//===- CombToCHALK.cpp - Comb To CHALK Conversion Pass --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Comb to CHALK Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToCHALK.h"
#include "../PassDetail.h"
#include "circt/Dialect/CHALK/CHALKOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace comb;
using namespace chalk;

namespace {

//===----------------------------------------------------------------------===//
// Expression Conversion
//===----------------------------------------------------------------------===//

struct AndOpConversion : public OpConversionPattern<AndOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AndOp op, OpAdaptor adaptor,
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
// Conversion Infrastructure
//===----------------------------------------------------------------------===//

static void populateLegality(ConversionTarget &target) {
  target.addLegalDialect<mlir::BuiltinDialect>();
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<chalk::CHALKDialect>();
}

static void populateTypeConversion(TypeConverter &typeConverter) {
}

static void populateOpConversion(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  auto *context = patterns.getContext();
  // clang-format off
  patterns.add<
    AndOpConversion
  >(typeConverter, context);
  // clang-format on
  mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      patterns, typeConverter);
}

//===----------------------------------------------------------------------===//
// Comb to CHALK Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct CombToCHALKPass : public CombToCHALKBase<CombToCHALKPass> {
  void runOnOperation() override;
};
} // namespace

/// Create a Comb to core dialects conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertCombToCHALKPass() {
  return std::make_unique<CombToCHALKPass>();
}

/// This is the main entrypoint for the Comb to CHALK conversion pass.
void CombToCHALKPass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(context);
  TypeConverter typeConverter;
  RewritePatternSet patterns(&context);
  populateLegality(target);
  populateTypeConversion(typeConverter);
  populateOpConversion(patterns, typeConverter);

  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
