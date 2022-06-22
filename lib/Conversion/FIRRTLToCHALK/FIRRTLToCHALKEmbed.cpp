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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

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
struct FIRRTLAndEmbed : public OpConversionPattern<AndPrimOp> {
  using OpConversionPattern::OpConversionPattern;
  // FIRRTLAndEmbed(MLIRContext *context)
  //     : OpConversionPattern<AndPrimOp>(context) {}

  LogicalResult
  matchAndRewrite(AndPrimOp andOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
  LLVM_DEBUG(llvm::dbgs() << "===----- Rewriting AndPrimOp -----===" << "\n");
    SmallString<32> firrtlAndOutNameStr;
    llvm::raw_svector_ostream tmpFIRRTLStream(firrtlAndOutNameStr);
    andOp.result().print(tmpFIRRTLStream);
    auto firrtlAndName = tmpFIRRTLStream.str();
    auto andLoc = andOp.getLoc();

    auto newAndOp = andOp.clone();
  LLVM_DEBUG(llvm::dbgs() << "===----- Create CellOp(AndPrimOp) -----===" << "\n");
    auto andCell = rewriter.create<CellOp>(andLoc, firrtlAndName);
    auto andCellLoc = andCell.getLoc();

    OperationState state(andCellLoc, firrtlAndName);
    auto *region = state.addRegion();
    region->emplaceBlock();
    // OpBuilder builder(region);
    // builder.createBlock(region);

    // auto svReg = rewriter.create<sv::RegOp>(loc, reg.getResult().getType(),
    //                                         reg.nameAttr());
    // svReg->setDialectAttrs(reg->getDialectAttrs());

    // // If the seq::CompRegOp has an inner_sym attribute, set this for the
    // // sv::RegOp inner_sym attribute.
    // if (andOp.sym_name().hasValue())
    //   svReg.inner_symAttr(andOp.sym_nameAttr());

    // auto regVal = rewriter.create<sv::ReadInOutOp>(loc, svReg);
    // if (andOp.reset() && andOp.resetValue()) {
    //   rewriter.create<sv::AlwaysFFOp>(
    //       loc, sv::EventControl::AtPosEdge, andOp.clk(),
    //       ResetType::SyncReset, sv::EventControl::AtPosEdge, andOp.reset(),
    //       [&]() { rewriter.create<sv::PAssignOp>(loc, svReg, andOp.input());
    //       },
    //       [&]() {
    //         rewriter.create<sv::PAssignOp>(loc, svReg, andOp.resetValue());
    //       });
    // } else {
    //   rewriter.create<sv::AlwaysFFOp>(
    //       loc, sv::EventControl::AtPosEdge, andOp.clk(),
    //       [&]() { rewriter.create<sv::PAssignOp>(loc, svReg, andOp.input());
    //       });
    // }

    rewriter.replaceOp(andOp, {newAndOp});
    return success();
  }
};

struct FIRRTLToCHALKEmbedPass
    : public FIRRTLToCHALKEmbedBase<FIRRTLToCHALKEmbedPass> {
  void runOnOperation() override;
};
} // namespace

std::unique_ptr<mlir::Pass> circt::createConvertFIRRTLToCHALKEmbedPass() {
  return std::make_unique<FIRRTLToCHALKEmbedPass>();
}

void FIRRTLToCHALKEmbedPass::runOnOperation() {
  auto &ctx = getContext();
  TypeConverter typeConverter;
  ConversionTarget target(ctx);
  target.addLegalDialect<firrtl::FIRRTLDialect>();
  target.addLegalDialect<chalk::CHALKDialect>();
  target.addLegalOp<CellOp>();
  target.addLegalOp<AndPrimOp>();
  target.markOpRecursivelyLegal<CellOp, AndPrimOp>();

  RewritePatternSet patterns(&ctx);
  patterns.add<FIRRTLAndEmbed>(&ctx);

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
