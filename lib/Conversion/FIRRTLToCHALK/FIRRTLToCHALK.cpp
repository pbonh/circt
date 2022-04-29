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

#include "circt/Conversion/FIRRTLToCHALK.h"
#include "../PassDetail.h"
#include "circt/Dialect/CHALK/CHALKOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace firrtl;
using namespace chalk;

namespace {

//===----------------------------------------------------------------------===//
// Expression Conversion
//===----------------------------------------------------------------------===//

struct AndPrimOpConversion : public OpConversionPattern<AndPrimOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AndPrimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // rewriter.replaceOpWithNewOp<chalk::CellOP>(op, adaptor.getValues());
    OperationState state(op.getLoc(), StringRef(""));
    Region *region = state.addRegion();
    Block *body = new Block();
    region->push_back(body);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// FIRRTL to CHALK Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct FIRRTLCHALKEmbed;

/// This is state shared across the parallel module embedding logic.
struct CircuitEmbeddingState {
};

struct FIRRTLCHALKEmbed : public FIRRTLVisitor<FIRRTLCHALKEmbed, LogicalResult> {
  FIRRTLCHALKEmbed(firrtl::FModuleOp module, CircuitEmbeddingState &circuitState)
      : circuitState(circuitState),
        builder(module.getLoc(), module.getContext()) {}

  LogicalResult run();

private:
  FModuleOp firrtlModule;
  CircuitEmbeddingState &circuitState;
  ImplicitLocOpBuilder builder;
};
} // namespace

LogicalResult FIRRTLCHALKEmbed::run() {
  for (auto &moduleOps : firrtlModule) {
  }
  // OperationState state(firrtlModule.getLoc(), firrtlModule->getName());
  // Region *region = state.addRegion();
  // Block *body = new Block();
  // region->push_back(body);
  return success();
}

namespace {
struct FIRRTLToCHALKPass : public FIRRTLToCHALKBase<FIRRTLToCHALKPass> {
  void runOnOperation() override;
};
} // namespace

/// Create a FIRRTL to CHALK conversion(embedding) pass.
std::unique_ptr<OperationPass<ModuleOp>>
circt::createConvertFIRRTLToCHALKPass() {
  return std::make_unique<FIRRTLToCHALKPass>();
}

/// This is the main entrypoint for the FIRRTL to CHALK conversion pass.
void FIRRTLToCHALKPass::runOnOperation() {
  auto *topLevelModule = getOperation().getBody();

  // Find the single firrtl.circuit in the module.
  CircuitOp circuit;
  for (auto &op : *topLevelModule) {
    if ((circuit = dyn_cast<CircuitOp>(&op)))
      break;
  }

  if (!circuit)
    return;

  auto *circuitBody = circuit.getBody();
  SmallVector<FModuleOp, 32> modulesToProcess;
  CircuitEmbeddingState state;

  for (auto &op : make_early_inc_range(circuitBody->getOperations())) {
    TypeSwitch<Operation *>(&op)
        .Case<FModuleOp>([&](auto module) {
          modulesToProcess.push_back(module);
        })
        .Default([&](Operation *op) {
        });
  }

  auto result = mlir::failableParallelForEachN(
      &getContext(), 0, modulesToProcess.size(), [&](auto index) {
        return FIRRTLCHALKEmbed(modulesToProcess[index], state).run();
      });

  if (failed(result))
    return signalPassFailure();

}
