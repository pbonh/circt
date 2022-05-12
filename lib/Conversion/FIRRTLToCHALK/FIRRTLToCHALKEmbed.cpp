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
struct FIRRTLToCHALKEmbedPass : public FIRRTLToCHALKEmbedBase<FIRRTLToCHALKEmbedPass> {
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
circt::createConvertFIRRTLToCHALKEmbedPass() {
  return std::make_unique<FIRRTLToCHALKEmbedPass>();
}

void FIRRTLToCHALKEmbedPass::runOnOperation() {
    auto *ctx = &getContext();
    TypeConverter typeConverter;
}

