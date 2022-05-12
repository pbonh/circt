//===- FIRRTLToCHALK.cpp - FIRRTL To CHALK Conversion Pass
//--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main FIRRTL to CHALK Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "circt/Conversion/FIRRTLToCHALK.h"
#include "circt/Dialect/CHALK/CHALKOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/Namespace.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "firrtl-to-chalk"

using namespace mlir;
using namespace circt;
using namespace firrtl;
using namespace chalk;


static Type lowerType(Type type) {
  auto firType = type.dyn_cast<FIRRTLType>();
  if (!firType)
    return {};

  // Ignore flip types.
  firType = firType.getPassiveType();

  if (BundleType bundle = firType.dyn_cast<BundleType>()) {
    mlir::SmallVector<hw::StructType::FieldInfo, 8> hwfields;
    for (auto element : bundle) {
      Type etype = lowerType(element.type);
      if (!etype)
        return {};
      hwfields.push_back(hw::StructType::FieldInfo{element.name, etype});
    }
    return hw::StructType::get(type.getContext(), hwfields);
  }
  if (FVectorType vec = firType.dyn_cast<FVectorType>()) {
    auto elemTy = lowerType(vec.getElementType());
    if (!elemTy)
      return {};
    return hw::ArrayType::get(elemTy, vec.getNumElements());
  }

  auto width = firType.getBitWidthOrSentinel();
  if (width >= 0) // IntType, analog with known width, clock, etc.
    return IntegerType::get(type.getContext(), width);

  return {};
}

static Value castToFIRRTLType(Value val, Type type,
                              ImplicitLocOpBuilder &builder) {
  auto firType = type.cast<FIRRTLType>();

  // Use HWStructCastOp for a bundle type.
  if (BundleType bundle = type.dyn_cast<BundleType>())
    val = builder.createOrFold<HWStructCastOp>(firType.getPassiveType(), val);

  if (type != val.getType())
    val = builder.create<mlir::UnrealizedConversionCastOp>(firType, val)
              .getResult(0);

  return val;
}

static Value castFromFIRRTLType(Value val, Type type,
                                ImplicitLocOpBuilder &builder) {

  if (hw::StructType structTy = type.dyn_cast<hw::StructType>()) {
    // Strip off Flip type if needed.
    val = builder
              .create<mlir::UnrealizedConversionCastOp>(
                  val.getType().cast<FIRRTLType>().getPassiveType(), val)
              .getResult(0);
    val = builder.createOrFold<HWStructCastOp>(type, val);
    return val;
  }

  val =
      builder.create<mlir::UnrealizedConversionCastOp>(type, val).getResult(0);

  return val;
}

static Value tryEliminatingAttachesToAnalogValue(Value value,
                                                 Operation *insertPoint) {
  if (!value.hasOneUse())
    return {};

  auto attach = dyn_cast<AttachOp>(*value.user_begin());
  if (!attach || attach.getNumOperands() != 2)
    return {};

  // Don't optimize zero bit analogs.
  auto loweredType = lowerType(value.getType());
  if (loweredType.isInteger(0))
    return {};

  // Check to see if the attached value dominates the insertion point.  If
  // not, just fail.
  auto attachedValue = attach.getOperand(attach.getOperand(0) == value);
  auto *op = attachedValue.getDefiningOp();
  if (op && op->getBlock() == insertPoint->getBlock() &&
      !op->isBeforeInBlock(insertPoint))
    return {};

  attach.erase();

  ImplicitLocOpBuilder builder(insertPoint->getLoc(), insertPoint);
  return castFromFIRRTLType(attachedValue, hw::InOutType::get(loweredType),
                            builder);
}

static Value tryEliminatingConnectsToValue(Value flipValue,
                                           Operation *insertPoint) {
  // Handle analog's separately.
  if (flipValue.getType().isa<AnalogType>())
    return tryEliminatingAttachesToAnalogValue(flipValue, insertPoint);

  Operation *connectOp = nullptr;
  for (auto &use : flipValue.getUses()) {
    // We only know how to deal with connects where this value is the
    // destination.
    if (use.getOperandNumber() != 0)
      return {};
    if (!isa<ConnectOp, StrictConnectOp>(use.getOwner()))
      return {};

    // We only support things with a single connect.
    if (connectOp)
      return {};
    connectOp = use.getOwner();
  }

  // We don't have an HW equivalent of "poison" so just don't special case
  // the case where there are no connects other uses of an output.
  if (!connectOp)
    return {}; // TODO: Emit an sv.constant here since it is unconnected.

  // Don't special case zero-bit results.
  auto loweredType = lowerType(flipValue.getType());
  if (loweredType.isInteger(0))
    return {};

  // Convert each connect into an extended version of its operand being
  // output.
  ImplicitLocOpBuilder builder(insertPoint->getLoc(), insertPoint);

  auto connectSrc = connectOp->getOperand(1);

  // Convert fliped sources to passive sources.
  if (!connectSrc.getType().cast<FIRRTLType>().isPassive())
    connectSrc =
        builder
            .create<mlir::UnrealizedConversionCastOp>(
                connectSrc.getType().cast<FIRRTLType>().getPassiveType(),
                connectSrc)
            .getResult(0);

  // We know it must be the destination operand due to the types, but the
  // source may not match the destination width.
  auto destTy = flipValue.getType().cast<FIRRTLType>().getPassiveType();

  if (!destTy.isGround()) {
    // If types are not ground type and they don't match, we give up.
    if (destTy != connectSrc.getType().cast<FIRRTLType>())
      return {};
  } else if (destTy.getBitWidthOrSentinel() !=
             connectSrc.getType().cast<FIRRTLType>().getBitWidthOrSentinel()) {
    // The only type mismatchs we care about is due to integer width
    // differences.
    auto destWidth = destTy.getBitWidthOrSentinel();
    assert(destWidth != -1 && "must know integer widths");
    connectSrc = builder.createOrFold<PadPrimOp>(destTy, connectSrc, destWidth);
  }

  // Remove the connect and use its source as the value for the output.
  connectOp->erase();

  // Convert from FIRRTL type to builtin type.
  return castFromFIRRTLType(connectSrc, loweredType, builder);
}

//===----------------------------------------------------------------------===//
// FIRRTL to CHALK Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct CircuitLoweringState;
struct FIRRTLCHALKEmbed : public FIRRTLVisitor<FIRRTLCHALKEmbed, LogicalResult> {
  FIRRTLCHALKEmbed(hw::HWModuleOp module, CircuitLoweringState &circuitState, Location loc)
      : hwModule(module), circuitState(circuitState),
        embedBuilder(loc, module->getContext()) {}
        // embedBuilder(ImplicitLocOpBuilder::atBlockBegin(loc, module->getBlock())) {}

  using FIRRTLVisitor<FIRRTLCHALKEmbed, LogicalResult>::visitExpr;

  LogicalResult run();

  LogicalResult visitExpr(AndPrimOp op);

  LogicalResult visitUnhandledOp(Operation *op) { return failure(); }
  LogicalResult visitInvalidOp(Operation *op) { return failure(); }

private:
  hw::HWModuleOp hwModule;
  CircuitLoweringState &circuitState;
  ImplicitLocOpBuilder embedBuilder;
};
} // namespace

LogicalResult FIRRTLCHALKEmbed::run() {
  auto &hwModuleBody = hwModule.getBody();
  auto &hwModuleBodyBegin = hwModuleBody.front();
  hwModuleBodyBegin.recomputeOpOrder();
  for (auto &op : hwModuleBodyBegin.getOperations()) {
    SmallString<32> resultNameStr;
    llvm::raw_svector_ostream tmpStream(resultNameStr);
    op.getLoc()->print(tmpStream);
    LLVM_DEBUG(llvm::dbgs() << "===----- Iterate HW Module, Op Module Loc: " << tmpStream.str().data() << "-----===" << "\n");

    // embedBuilder.setInsertionPoint(&op);
    // embedBuilder.setLoc(op.getLoc());
    succeeded(dispatchVisitor(&op));
  }
  return success();
}

LogicalResult FIRRTLCHALKEmbed::visitExpr(AndPrimOp op) {
  LLVM_DEBUG(llvm::dbgs() << "===----- Visiting AndPrimOp -----===" << "\n");
  SmallString<32> firrtlAndOutNameStr;
  llvm::raw_svector_ostream tmpFIRRTLStream(firrtlAndOutNameStr);
  op.result().print(tmpFIRRTLStream);

  LLVM_DEBUG(llvm::dbgs() << "===----- Building CellOp for AndPrimOp -----===" << "\n");
  auto andCell = embedBuilder.create<CellOp, StringRef>(tmpFIRRTLStream.str());
  auto andLoc = andCell.getLoc();
  SmallString<32> locNameStr;
  llvm::raw_svector_ostream tmpStream(locNameStr);
  andLoc->print(tmpStream);
  LLVM_DEBUG(llvm::dbgs() << "===----- Building CellOp for AndPrimOp Loc: " << tmpStream.str().data() << "-----===" << "\n");

  // OperationState state(andLoc, firrtlAndName);
  // Region *region = state.addRegion();
  // OpBuilder builder(region);
  // auto *body = new Block();
  // // body->push_back(andCell);
  // region->push_back(body);

  return success();
}

namespace {
struct FIRRTLToCHALKPass;

struct CircuitLoweringState {
  CircuitLoweringState(CircuitOp circuitOp, InstanceGraph *instanceGraph)
      : circuitOp(circuitOp), instanceGraph(instanceGraph) { }

  CircuitOp circuitOp;

  Operation *getNewModule(Operation *oldModule) {
    auto it = oldToNewModuleMap.find(oldModule);
    return it != oldToNewModuleMap.end() ? it->second : nullptr;
  }

  InstanceGraph *getInstanceGraph() { return instanceGraph; }

private:
  friend struct FIRRTLToCHALKPass;
  CircuitLoweringState(const CircuitLoweringState &) = delete;
  void operator=(const CircuitLoweringState &) = delete;

  DenseMap<Operation *, Operation *> oldToNewModuleMap;
  InstanceGraph *instanceGraph;
};

struct FIRRTLToCHALKPass : public FIRRTLToCHALKBase<FIRRTLToCHALKPass> {
  void runOnOperation() override;
private:
  hw::HWModuleOp lowerModule(FModuleOp oldModule, Block *topLevelBlock);
  LogicalResult lowerPorts(ArrayRef<PortInfo> firrtlPorts,
                           SmallVectorImpl<hw::PortInfo> &ports,
                           Operation *moduleOp);
  LogicalResult lowerModuleBody(FModuleOp oldModule,
                                CircuitLoweringState &loweringState);
  LogicalResult lowerModuleOperations(hw::HWModuleOp module,
                                      CircuitLoweringState &loweringState,
                                      Location loc);
};
} // namespace

std::unique_ptr<mlir::Pass>
circt::createConvertFIRRTLToCHALKPass() {
  return std::make_unique<FIRRTLToCHALKPass>();
}

void FIRRTLToCHALKPass::runOnOperation() {
  auto *topLevelModule = getOperation().getBody();

  CircuitOp circuit;
  for (auto &op : *topLevelModule) {
    if ((circuit = dyn_cast<CircuitOp>(&op)))
      break;
  }

  if (!circuit)
    return;

  auto *circuitBody = circuit.getBody();
  auto &topLevelRegion = getOperation().getBodyRegion();
  auto &topLevelBlock = topLevelRegion.back();
  SmallVector<FModuleOp, 32> modulesToProcess;

  CircuitLoweringState state(circuit, &getAnalysis<InstanceGraph>());

  for (auto &op : make_early_inc_range(circuitBody->getOperations())) {
    TypeSwitch<Operation *>(&op)
        .Case<FModuleOp>([&](auto module) {
          modulesToProcess.push_back(module);
          auto loweredMod = lowerModule(module, &topLevelBlock);
          if (!loweredMod)
            return signalPassFailure();
          state.oldToNewModuleMap[&op] = loweredMod;
        })
        .Default([&](Operation *op) {
        });
  }

  auto result = mlir::failableParallelForEachN(
      &getContext(), 0, modulesToProcess.size(), [&](auto index) {
        return lowerModuleBody(modulesToProcess[index], state);
      });

  if (failed(result))
    return signalPassFailure();
}

hw::HWModuleOp
FIRRTLToCHALKPass::lowerModule(FModuleOp oldModule, Block *topLevelBlock) {
  SmallVector<PortInfo> firrtlPorts = oldModule.getPorts();
  SmallVector<hw::PortInfo, 8> ports;
  if (failed(lowerPorts(firrtlPorts, ports, oldModule)))
    return {};

  auto hwModuleInsertLoc = topLevelBlock->back().getLoc();
  auto builder = ImplicitLocOpBuilder::atBlockEnd(hwModuleInsertLoc, topLevelBlock);
  auto newModule = builder.create<hw::HWModuleOp>(oldModule.getNameAttr(), ports);

  // SymbolTable::setSymbolVisibility(newModule,
  //                                  SymbolTable::getSymbolVisibility(oldModule));

  return newModule;
}

LogicalResult FIRRTLToCHALKPass::lowerPorts(
    ArrayRef<PortInfo> firrtlPorts, SmallVectorImpl<hw::PortInfo> &ports,
    Operation *moduleOp) {
  ports.reserve(firrtlPorts.size());
  size_t numArgs = 0;
  size_t numResults = 0;
  for (auto firrtlPort : firrtlPorts) {
    hw::PortInfo hwPort;
    hwPort.name = firrtlPort.name;
    hwPort.type = lowerType(firrtlPort.type);
    hwPort.sym = firrtlPort.sym;

    if (!hwPort.type) {
      moduleOp->emitError("cannot lower this port type to HW");
      return failure();
    }

    if (hwPort.type.isInteger(0))
      continue;

    if (firrtlPort.isOutput()) {
      hwPort.direction = hw::PortDirection::OUTPUT;
      hwPort.argNum = numResults++;
    } else if (firrtlPort.isInput()) {
      hwPort.direction = hw::PortDirection::INPUT;
      hwPort.argNum = numArgs++;
    } else {
      hwPort.type = hw::InOutType::get(hwPort.type);
      hwPort.direction = hw::PortDirection::INOUT;
      hwPort.argNum = numArgs++;
    }
    ports.push_back(hwPort);
  }
  return success();
}

LogicalResult
FIRRTLToCHALKPass::lowerModuleBody(FModuleOp oldModule,
                                      CircuitLoweringState &loweringState) {
  auto newModule =
      dyn_cast_or_null<hw::HWModuleOp>(loweringState.getNewModule(oldModule));
  if (!newModule)
    return success();

  ImplicitLocOpBuilder bodyBuilder(newModule.getLoc(), newModule.body());

  auto cursor = bodyBuilder.create<hw::ConstantOp>(APInt(1, 1));
  bodyBuilder.setInsertionPoint(cursor);

  SmallVector<PortInfo> ports = oldModule.getPorts();
  assert(oldModule.body().getNumArguments() == ports.size() &&
         "port count mismatch");

  size_t nextNewArg = 0;
  size_t firrtlArg = 0;
  SmallVector<Value, 4> outputs;

  auto *outputOp = newModule.getBodyBlock()->getTerminator();
  ImplicitLocOpBuilder outputBuilder(newModule.getLoc(), outputOp);

  for (auto &port : ports) {
    auto oldArg = oldModule.body().getArgument(firrtlArg++);

    bool isZeroWidth =
        port.type.cast<FIRRTLType>().getBitWidthOrSentinel() == 0;

    if (!port.isOutput() && !isZeroWidth) {
      Value newArg = newModule.body().getArgument(nextNewArg++);

      newArg = castToFIRRTLType(newArg, oldArg.getType(), bodyBuilder);
      oldArg.replaceAllUsesWith(newArg);
      continue;
    }

    if (isZeroWidth && port.isInput()) {
      Value newArg = bodyBuilder.create<WireOp>(
          port.type, "." + port.getName().str() + ".0width_input");
      oldArg.replaceAllUsesWith(newArg);
      continue;
    }

    if (auto value = tryEliminatingConnectsToValue(oldArg, outputOp)) {
      outputs.push_back(value);
      assert(oldArg.use_empty() && "should have removed all uses of oldArg");
      continue;
    }

    Value newArg = bodyBuilder.create<WireOp>(
        port.type, "." + port.getName().str() + ".output");
    oldArg.replaceAllUsesWith(newArg);

    auto resultHWType = lowerType(port.type);
    if (!resultHWType.isInteger(0)) {
      auto output = castFromFIRRTLType(newArg, resultHWType, outputBuilder);
      outputs.push_back(output);
    }
  }

  outputOp->setOperands(outputs);

  auto &oldBlockInstList = oldModule.clone().getBody()->getOperations();
  auto newModuleInsertLoc = newModule.getBodyBlock()->front().getLoc();
  auto &newBlockInstList = newModule.getBodyBlock()->getOperations();
  newBlockInstList.splice(Block::iterator(cursor), oldBlockInstList,
                          oldBlockInstList.begin(), oldBlockInstList.end());

  cursor.erase();

  return lowerModuleOperations(newModule, loweringState, newModuleInsertLoc);
}

LogicalResult FIRRTLToCHALKPass::lowerModuleOperations(
    hw::HWModuleOp module, CircuitLoweringState &loweringState, Location loc) {
  return FIRRTLCHALKEmbed(module, loweringState, loc).run();
}
