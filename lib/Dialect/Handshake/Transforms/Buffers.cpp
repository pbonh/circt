//===- Buffers.cpp - buffer materialization passes --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of buffer materialization passes.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace handshake;
using namespace mlir;

namespace {

struct RemoveHandshakeBuffers : public OpRewritePattern<handshake::BufferOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::BufferOp bufferOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(bufferOp, bufferOp.operand());
    return success();
  }
};

struct HandshakeRemoveBuffersPass
    : public HandshakeRemoveBuffersBase<HandshakeRemoveBuffersPass> {
  void runOnOperation() override {
    handshake::FuncOp op = getOperation();
    ConversionTarget target(getContext());
    target.addIllegalOp<handshake::BufferOp>();
    RewritePatternSet patterns(&getContext());
    patterns.insert<RemoveHandshakeBuffers>(&getContext());

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  };
};

struct HandshakeInsertBuffersPass
    : public HandshakeInsertBuffersBase<HandshakeInsertBuffersPass> {

  // Returns true if a block argument should have buffers added to its uses.
  static bool shouldBufferArgument(BlockArgument arg) {
    // At the moment, buffers only make sense on arguments which we know
    // will lower down to a handshake bundle.
    return arg.getType().isIntOrFloat() || arg.getType().isa<NoneType>();
  }

  static bool isUnbufferedChannel(Operation *definingOp, Operation *usingOp) {
    return !isa_and_nonnull<BufferOp>(definingOp) && !isa<BufferOp>(usingOp);
  }

  void insertBuffer(Location loc, Value operand, OpBuilder &builder,
                    unsigned numSlots, BufferTypeEnum bufferType) {
    auto ip = builder.saveInsertionPoint();
    builder.setInsertionPointAfterValue(operand);
    auto bufferOp = builder.create<handshake::BufferOp>(
        loc, operand.getType(), numSlots, operand, bufferType);
    operand.replaceUsesWithIf(
        bufferOp,
        function_ref<bool(OpOperand &)>([](OpOperand &operand) -> bool {
          return !isa<handshake::BufferOp>(operand.getOwner());
        }));
    builder.restoreInsertionPoint(ip);
  }

  // Inserts a buffer at a specific operand use.
  void bufferOperand(OpOperand &use, OpBuilder &builder, size_t numSlots,
                     BufferTypeEnum bufferType) {
    auto *usingOp = use.getOwner();
    Value usingValue = use.get();

    builder.setInsertionPoint(usingOp);
    auto buffer = builder.create<handshake::BufferOp>(
        usingValue.getLoc(), usingValue.getType(),
        /*slots=*/numSlots, usingValue,
        /*bufferType=*/bufferType);
    usingOp->setOperand(use.getOperandNumber(), buffer);
  }

  // Inserts buffers at all operands of an operation.
  void bufferOperands(Operation *op, OpBuilder builder, size_t numSlots,
                      BufferTypeEnum bufferType) {
    for (auto &use : op->getOpOperands()) {
      auto *srcOp = use.get().getDefiningOp();
      if (isa_and_nonnull<handshake::BufferOp>(srcOp))
        continue;
      bufferOperand(use, builder, numSlots, bufferType);
    }
  }

  // Inserts buffers at all results of an operation
  void bufferResults(OpBuilder &builder, Operation *op, unsigned numSlots,
                     BufferTypeEnum bufferType) {
    for (auto res : op->getResults()) {
      Operation *user = *res.getUsers().begin();
      if (isa<handshake::BufferOp>(user))
        continue;
      insertBuffer(op->getLoc(), res, builder, numSlots, bufferType);
    }
  };

  // Add a buffer to any un-buffered channel.
  void bufferAllStrategy(handshake::FuncOp f, OpBuilder &builder,
                         unsigned numSlots,
                         BufferTypeEnum bufferType = BufferTypeEnum::seq) {

    for (auto &arg : f.getArguments()) {
      if (!shouldBufferArgument(arg))
        continue;
      insertBuffer(arg.getLoc(), arg, builder, numSlots, bufferType);
    }

    for (auto &defOp : f.getOps()) {
      for (auto res : defOp.getResults()) {
        for (auto *useOp : res.getUsers()) {
          if (!isUnbufferedChannel(&defOp, useOp))
            continue;
          insertBuffer(res.getLoc(), res, builder, numSlots, bufferType);
        }
      }
    }
  }

  // Combination of bufferCyclesStrategy and bufferAllStrategy, where we add a
  // sequential buffer on graph cycles, and add FIFO buffers on all other
  // connections.
  void bufferAllFIFOStrategy(handshake::FuncOp f, OpBuilder &builder) {
    // First, buffer cycles with sequential buffers
    bufferCyclesStrategy(f, builder, /*numSlots=*/2,
                         /*bufferType=*/BufferTypeEnum::seq);
    // Then, buffer remaining channels with transparent FIFO buffers
    bufferAllStrategy(f, builder, bufferSize,
                      /*bufferType=*/BufferTypeEnum::fifo);
  }

  // Perform a depth first search and insert buffers when cycles are detected.
  void
  bufferCyclesStrategy(handshake::FuncOp f, OpBuilder &builder,
                       unsigned numSlots,
                       BufferTypeEnum /*bufferType*/ = BufferTypeEnum::seq) {
    // Cycles can only occur at merge-like operations so those are our buffering
    // targets. Placing the buffer at the output of the merge-like op,
    // as opposed to naivly placing buffers *whenever* cycles are detected
    // ensures that we don't place a bunch of buffers on each input of the
    // merge-like op.
    auto isSeqBuffer = [](auto op) {
      auto bufferOp = dyn_cast<handshake::BufferOp>(op);
      return bufferOp && bufferOp.isSequential();
    };

    for (auto mergeOp : f.getOps<MergeLikeOpInterface>()) {
      // We insert a sequential buffer whenever the op is determined to be
      // within a cycle (to break combinational cycles). Else, place a FIFO
      // buffer.
      bool sequential = inCycle(mergeOp, isSeqBuffer);
      bufferResults(builder, mergeOp, numSlots,
                    sequential ? BufferTypeEnum::seq : BufferTypeEnum::fifo);
    }
  }

  // Returns true if 'src' is within a cycle. 'breaksCycle' is a function which
  // determines whether an operation breaks a cycle.
  bool inCycle(Operation *src,
               llvm::function_ref<bool(Operation *)> breaksCycle,
               Operation *curr = nullptr, SetVector<Operation *> path = {}) {
    // If visiting the source node, then we're in a cycle.
    if (curr == src)
      return true;

    // Initial case; set current node to source node
    if (curr == nullptr) {
      curr = src;
    }

    path.insert(curr);
    for (auto &operand : curr->getUses()) {
      auto *user = operand.getOwner();

      // We might encounter a cycle, but we only care about the case when such
      // cycles include 'src'.
      if (path.count(user) && user != src)
        continue;
      if (breaksCycle(curr))
        continue;
      if (inCycle(src, breaksCycle, user, path))
        return true;
    }
    return false;
  }

  void runOnOperation() override {
    auto f = getOperation();
    auto builder = OpBuilder(f.getContext());

    if (strategy == "cycles")
      bufferCyclesStrategy(f, builder, bufferSize);
    else if (strategy == "all")
      bufferAllStrategy(f, builder, bufferSize);
    else if (strategy == "allFIFO")
      bufferAllFIFOStrategy(f, builder);
    else {
      getOperation().emitOpError() << "Unknown buffer strategy: " << strategy;
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
circt::handshake::createHandshakeRemoveBuffersPass() {
  return std::make_unique<HandshakeRemoveBuffersPass>();
}

std::unique_ptr<mlir::OperationPass<handshake::FuncOp>>
circt::handshake::createHandshakeInsertBuffersPass() {
  return std::make_unique<HandshakeInsertBuffersPass>();
}
