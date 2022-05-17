//===- HWToCHALK.cpp - HW To CHALK Conversion Pass
//--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main HW to CHALK Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "circt/Conversion/HWToCHALK.h"
#include "circt/Dialect/CHALK/CHALKOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/Namespace.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "hw-to-chalk-embed"

using namespace mlir;
using namespace circt;
using namespace hw;
using namespace chalk;

//===----------------------------------------------------------------------===//
// HW to CHALK Conversion Pass
//===----------------------------------------------------------------------===//

namespace {

struct HWEmbedPatternRewriter : public PatternRewriter {
  HWEmbedPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
};

struct HWOrEmbed : public RewritePattern {
  using RewritePattern::RewritePattern;
  HWOrEmbed(MLIRContext *ctx)
      : RewritePattern(comb::OrOp::getOperationName(), PatternBenefit(1), ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);
    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

struct HWAndEmbed : public OpConversionPattern<comb::AndOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comb::AndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);
    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

struct HWToCHALKEmbedPass : public HWToCHALKEmbedBase<HWToCHALKEmbedPass> {
  void runOnOperation() override;
};
} // namespace

std::unique_ptr<mlir::Pass> circt::createConvertHWToCHALKEmbedPass() {
  return std::make_unique<HWToCHALKEmbedPass>();
}

void HWToCHALKEmbedPass::runOnOperation() {
  auto *ctx = &getContext();

  // TypeConverter typeConverter;
  // HWEmbedPatternRewriter rewriter(ctx);
  ConversionTarget target(*ctx);
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<chalk::CHALKDialect>();
  target.addLegalOp<CellOp>();
  target.addLegalOp<comb::AndOp>();
  target.addLegalOp<comb::OrOp>();
  target.markOpRecursivelyLegal<CellOp, comb::AndOp, comb::OrOp>();

  RewritePattern::create<HWAndEmbed>(ctx);
  RewritePattern::create<HWOrEmbed>(ctx);

  RewritePatternSet patterns(ctx);
  patterns.addWithLabel<HWAndEmbed>({"HW Synth: AND"}, ctx);
  patterns.addWithLabel<HWOrEmbed>({"HW Synth: OR"}, ctx);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
