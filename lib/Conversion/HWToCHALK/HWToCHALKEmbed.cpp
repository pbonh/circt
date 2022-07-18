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
using namespace comb;

//===----------------------------------------------------------------------===//
// HW to CHALK Conversion Pass
//===----------------------------------------------------------------------===//

namespace {

struct HWEmbedPatternRewriter : public PatternRewriter {
  HWEmbedPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
};

// struct HWCHALKEmbedConversion : public ConversionPattern {
//   using ConversionPattern::ConversionPattern;
//   LogicalResult
//   matchAndRewrite(Operation *op, ArrayRef<Value> operands,
//                   ConversionPatternRewriter &rewriter) const override {
//   }
// };

struct HWCHALKEmbedRewrite : public RewritePattern {
  // using RewritePattern::RewritePattern;
  HWCHALKEmbedRewrite(MLIRContext *context)
   : RewritePattern(comb::AndOp::getOperationName(), PatternBenefit(1), context) {}

  // HWCHALKEmbedRewrite()
  //  : RewritePattern(PatternBenefit(1), MatchAnyOpTypeTag()) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) {
    if (isa<comb::AndOp>(op)) {
      auto andOp = dyn_cast<comb::AndOp>(op);
      LLVM_DEBUG(llvm::dbgs() << "===----- Rewriting AndPrimOp -----==="
                              << "\n");
      SmallString<32> firrtlAndOutNameStr;
      llvm::raw_svector_ostream tmpFIRRTLStream(firrtlAndOutNameStr);
      andOp.result().print(tmpFIRRTLStream);
      auto firrtlAndName = tmpFIRRTLStream.str();
      auto andLoc = andOp.getLoc();
      auto andCell = rewriter.create<CellOp>(andLoc, firrtlAndName);
      auto andCellLoc = andCell.getLoc();
      OperationState state(andCellLoc, firrtlAndName);
      auto *region = state.addRegion();
      auto &andCellOpBlock = region->emplaceBlock();
    }
  }
};

struct HWOrEmbed : public OpRewritePattern<comb::OrOp> {
  using OpRewritePattern::OpRewritePattern;
  // HWOrEmbed(MLIRContext *ctx)
  //     : OpRewritePattern<comb::OrOp>(ctx, PatternBenefit(1)) {}

  LogicalResult matchAndRewrite(comb::OrOp op,
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

std::unique_ptr<Pass> circt::createConvertHWToCHALKEmbedPass() {
  return std::make_unique<HWToCHALKEmbedPass>();
}

using OpList = SmallVector<Operation *>;

void HWToCHALKEmbedPass::runOnOperation() {
  MLIRContext &ctx = getContext();
  ModuleOp module = getOperation();
  /*
  OpBuilder builder(&ctx);
  OpList hwModules;
  llvm::for_each(getOperation().getBody()->getOperations(), [&](Operation &op) {
    if (isa<comb::AndOp>(op)) {
      hwModules.push_back(&op);
    }
  });

  for (auto *op : hwModules ) {
    auto hwModulesOps = op->getBody()->getOperations();
    for (auto *op : hwModules ) {
      auto andOp = dyn_cast<comb::AndOp>(op);

      SmallString<32> firrtlAndOutNameStr;
      llvm::raw_svector_ostream tmpFIRRTLStream(firrtlAndOutNameStr);
      andOp.result().print(tmpFIRRTLStream);
      auto firrtlAndName = tmpFIRRTLStream.str();
      auto andLoc = andOp.getLoc();

      OperationState state(andLoc, firrtlAndName);
      auto *region = state.addRegion();
      auto &andOpBlock = region->emplaceBlock();

      OpBuilder builder(&andOpBlock, andOpBlock.begin());
      CellOp::build(builder, state, firrtlAndName);
    }
  }
  */

  // TypeConverter typeConverter;
  // HWEmbedPatternRewriter rewriter(ctx);

  // RewritePattern::create<HWAndEmbed>(ctx);
  // RewritePattern::create<HWOrEmbed>(ctx);

  ConversionTarget target(ctx);
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<chalk::CHALKDialect>();
  target.addLegalOp<hw::HWModuleOp>();
  target.addLegalOp<CellOp>();
  target.addLegalOp<comb::AndOp>();
  target.addLegalOp<comb::OrOp>();
  target.markOpRecursivelyLegal<hw::HWModuleOp, CellOp, comb::AndOp, comb::OrOp>();

  RewritePatternSet patterns(&ctx);
  // patterns.add<HWAndEmbed>(&ctx);
  // patterns.add<HWOrEmbed>(&ctx);
  patterns.add<HWCHALKEmbedRewrite>(&ctx);
  // patterns.add<HWCHALKEmbedConversion>(&ctx);

  if (failed(
          applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
