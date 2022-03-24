//===- FlattenMemRefs.cpp - MemRef flattening pass --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the MemRef flattening pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace circt;

static bool isUniDimensional(MemRefType memref) {
  return memref.getShape().size() == 1;
}

/// A struct for maintaining function declarations which needs to be rewritten,
/// if they contain memref arguments that was flattened.
struct FunctionRewrite {
  FuncOp op;
  FunctionType type;
};

// Flatten indices by generating the product of the i'th index and the [0:i-1]
// shapes, for each index, and then summing these.
static Value flattenIndices(ConversionPatternRewriter &rewriter, Operation *op,
                            ValueRange indices, MemRefType memrefType) {
  assert(memrefType.hasStaticShape() && "expected statically shaped memref");
  Location loc = op->getLoc();
  Value finalIdx = indices.front();
  for (auto memIdx : llvm::enumerate(indices.drop_front())) {
    Value partialIdx = memIdx.value();
    int64_t indexMulFactor = 1;

    // Calculate the product of the i'th index and the [0:i-1] shape dims.
    for (unsigned i = 0; i <= memIdx.index(); ++i) {
      int64_t dimSize = memrefType.getShape()[i];
      indexMulFactor *= dimSize;
    }

    // Multiply product by the current index operand.
    if (llvm::isPowerOf2_64(indexMulFactor)) {
      auto constant =
          rewriter
              .create<arith::ConstantOp>(
                  loc, rewriter.getIndexAttr(llvm::Log2_64(indexMulFactor)))
              .getResult();
      partialIdx =
          rewriter.create<arith::ShLIOp>(loc, partialIdx, constant).getResult();
    } else {
      auto constant = rewriter
                          .create<arith::ConstantOp>(
                              loc, rewriter.getIndexAttr(indexMulFactor))
                          .getResult();
      partialIdx =
          rewriter.create<arith::MulIOp>(loc, partialIdx, constant).getResult();
    }

    // Sum up with the prior lower dimension accessors.
    auto sumOp = rewriter.create<arith::AddIOp>(loc, finalIdx, partialIdx);
    finalIdx = sumOp.getResult();
  }
  return finalIdx;
}

static bool hasMultiDimMemRef(ValueRange values) {
  return llvm::any_of(values, [](Value v) {
    auto memref = v.getType().dyn_cast<MemRefType>();
    if (!memref)
      return false;
    return !isUniDimensional(memref);
  });
}

namespace {

struct LoadOpConversion : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = op.getMemRefType();
    if (isUniDimensional(type) || !type.hasStaticShape() ||
        /*Already converted?*/ op.getIndices().size() == 1)
      return failure();
    Value finalIdx =
        flattenIndices(rewriter, op, adaptor.indices(), op.getMemRefType());
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.memref(),

                                                SmallVector<Value>{finalIdx});
    return success();
  }
};

struct StoreOpConversion : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = op.getMemRefType();
    if (isUniDimensional(type) || !type.hasStaticShape() ||
        /*Already converted?*/ op.getIndices().size() == 1)
      return failure();
    Value finalIdx =
        flattenIndices(rewriter, op, adaptor.indices(), op.getMemRefType());
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, adaptor.value(), adaptor.memref(), SmallVector<Value>{finalIdx});
    return success();
  }
};

struct AllocOpConversion : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = op.getType();
    if (isUniDimensional(type) || !type.hasStaticShape())
      return failure();
    MemRefType newType = MemRefType::get(
        SmallVector<int64_t>{type.getNumElements()}, type.getElementType());
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, newType);
    return success();
  }
};

struct ReturnOpConversion : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.operands());
    return success();
  }
};

struct CondBranchOpConversion
    : public OpConversionPattern<mlir::cf::CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::cf::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        op, adaptor.getCondition(), adaptor.getTrueDestOperands(),
        adaptor.getFalseDestOperands(), op.getTrueDest(), op.getFalseDest());
    return success();
  }
};

struct BranchOpConversion : public OpConversionPattern<mlir::cf::BranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::cf::BranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, op.getDest(),
                                                    adaptor.getDestOperands());
    return success();
  }
};

// Rewrites a call op signature to flattened types. If rewriteFunctions is set,
// will also replace the callee with a private definition of the called
// function of the updated signature.
struct CallOpConversion : public OpConversionPattern<func::CallOp> {
  CallOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   bool rewriteFunctions = false)
      : OpConversionPattern(typeConverter, context),
        rewriteFunctions(rewriteFunctions) {}

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Type> convResTypes;
    if (typeConverter->convertTypes(op.getResultTypes(), convResTypes).failed())
      return failure();
    auto newCallOp = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, adaptor.getCallee(), convResTypes, adaptor.getOperands());

    if (!rewriteFunctions)
      return success();

    // Override any definition corresponding to the updated signature.
    // It is up to users of this pass to define how these rewritten functions
    // are to be implemented.
    rewriter.setInsertionPoint(op->getParentOfType<FuncOp>());
    auto *calledFunction = dyn_cast<CallOpInterface>(*op).resolveCallable();
    FunctionType funcType = FunctionType::get(
        op.getContext(), newCallOp.getOperandTypes(), convResTypes);
    FuncOp newFuncOp;
    if (calledFunction)
      newFuncOp = rewriter.replaceOpWithNewOp<FuncOp>(calledFunction,
                                                      op.getCallee(), funcType);
    else
      newFuncOp =
          rewriter.create<FuncOp>(op.getLoc(), op.getCallee(), funcType);
    newFuncOp.setVisibility(SymbolTable::Visibility::Private);

    return success();
  }

private:
  bool rewriteFunctions;
};

template <typename TOp>
void addGenericLegalityConstraint(ConversionTarget &target) {
  target.addDynamicallyLegalOp<TOp>([](TOp op) {
    return !hasMultiDimMemRef(op->getOperands()) &&
           !hasMultiDimMemRef(op->getResults());
  });
}

static void populateFlattenMemRefsLegality(ConversionTarget &target) {
  target.addLegalDialect<arith::ArithmeticDialect>();
  target.addDynamicallyLegalOp<memref::AllocOp>(
      [](memref::AllocOp op) { return isUniDimensional(op.getType()); });
  target.addDynamicallyLegalOp<memref::StoreOp>(
      [](memref::StoreOp op) { return op.getIndices().size() == 1; });
  target.addDynamicallyLegalOp<memref::LoadOp>(
      [](memref::LoadOp op) { return op.getIndices().size() == 1; });

  addGenericLegalityConstraint<mlir::cf::CondBranchOp>(target);
  addGenericLegalityConstraint<mlir::cf::BranchOp>(target);
  addGenericLegalityConstraint<func::CallOp>(target);
  addGenericLegalityConstraint<func::ReturnOp>(target);

  target.addDynamicallyLegalOp<FuncOp>([](FuncOp op) {
    auto argsConverted = llvm::none_of(op.getBlocks(), [](auto &block) {
      return hasMultiDimMemRef(block.getArguments());
    });

    auto resultsConverted = llvm::all_of(op.getResultTypes(), [](Type type) {
      if (auto memref = type.dyn_cast<MemRefType>())
        return isUniDimensional(memref);
      return true;
    });

    return argsConverted && resultsConverted;
  });
}

// Materializes a multidimensional memory to unidimensional memory by using a
// memref.subview operation.
// TODO: This is also possible for dynamically shaped memories.
static Value materializeSubViewFlattening(OpBuilder &builder, MemRefType type,
                                          ValueRange inputs, Location loc) {
  assert(type.hasStaticShape() &&
         "Can only subview flatten memref's with static shape (for now...).");
  MemRefType sourceType = inputs[0].getType().cast<MemRefType>();
  int64_t memSize = sourceType.getNumElements();
  unsigned dims = sourceType.getShape().size();

  // Build offset, sizes and strides
  SmallVector<OpFoldResult> sizes(dims, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> offsets(dims, builder.getIndexAttr(1));
  offsets[offsets.size() - 1] = builder.getIndexAttr(memSize);
  SmallVector<OpFoldResult> strides(dims, builder.getIndexAttr(1));

  // Generate the appropriate return type:
  MemRefType outType = MemRefType::get({memSize}, type.getElementType());
  return builder.create<memref::SubViewOp>(loc, outType, inputs[0], sizes,
                                           offsets, strides);
}

static void populateTypeConversionPatterns(TypeConverter &typeConverter) {
  // Add default conversion for all types generically.
  typeConverter.addConversion([](Type type) { return type; });
  // Add specific conversion for memref types.
  typeConverter.addConversion([](MemRefType memref) {
    if (isUniDimensional(memref))
      return memref;
    return MemRefType::get(llvm::SmallVector<int64_t>{memref.getNumElements()},
                           memref.getElementType());
  });
}

struct FlattenMemRefPass : public FlattenMemRefBase<FlattenMemRefPass> {
public:
  void runOnOperation() override {

    auto *ctx = &getContext();
    TypeConverter typeConverter;
    populateTypeConversionPatterns(typeConverter);

    RewritePatternSet patterns(ctx);
    SetVector<StringRef> rewrittenCallees;
    patterns.add<LoadOpConversion, StoreOpConversion, AllocOpConversion,
                 ReturnOpConversion, CondBranchOpConversion, BranchOpConversion,
                 CallOpConversion>(typeConverter, ctx);
    populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                             typeConverter);

    ConversionTarget target(*ctx);
    populateFlattenMemRefsLegality(target);

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed()) {
      signalPassFailure();
      return;
    }
  }
};

struct FlattenMemRefCallsPass
    : public FlattenMemRefCallsBase<FlattenMemRefCallsPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    TypeConverter typeConverter;
    populateTypeConversionPatterns(typeConverter);
    RewritePatternSet patterns(ctx);

    // Only run conversion on call ops within the body of the function. callee
    // functions are rewritten by rewriteFunctions=true. We do not use
    // populateFuncOpTypeConversionPattern to rewrite the function signatures,
    // since non-called functions should not have their types converted.
    // It is up to users of this pass to define how these rewritten functions
    // are to be implemented.
    patterns.add<CallOpConversion>(typeConverter, ctx,
                                   /*rewriteFunctions=*/true);

    ConversionTarget target(*ctx);
    target.addLegalDialect<memref::MemRefDialect, mlir::BuiltinDialect>();
    addGenericLegalityConstraint<func::CallOp>(target);

    // Add a target materializer to handle memory flattening through
    // memref.subview operations.
    typeConverter.addTargetMaterialization(materializeSubViewFlattening);

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed()) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace circt {
std::unique_ptr<mlir::Pass> createFlattenMemRefPass() {
  return std::make_unique<FlattenMemRefPass>();
}

std::unique_ptr<mlir::Pass> createFlattenMemRefCallsPass() {
  return std::make_unique<FlattenMemRefCallsPass>();
}

} // namespace circt
