//===- CHALKOps.h - Declare CHALK dialect operations --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the CHALK dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CHALK_CHALKOPS_H
#define CIRCT_DIALECT_CHALK_CHALKOPS_H

#include "circt/Dialect/CHALK/CHALKDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace llvm {
struct KnownBits;
}

namespace mlir {
class PatternRewriter;
}

#define GET_OP_CLASSES
#include "circt/Dialect/CHALK/CHALK.h.inc"

namespace circt {
namespace chalk {

using llvm::KnownBits;

/// Compute "known bits" information about the specified value - the set of bits
/// that are guaranteed to always be zero, and the set of bits that are
/// guaranteed to always be one (these must be exclusive!).  A bit that exists
/// in neither set is unknown.
KnownBits computeKnownBits(Value value);

/// Create a sign extension operation from a value of integer type to an equal
/// or larger integer type.
Value createOrFoldSExt(Location loc, Value value, Type destTy,
                       OpBuilder &builder);
Value createOrFoldSExt(Value value, Type destTy, ImplicitLocOpBuilder &builder);

/// Create a ``Not'' gate on a value.
Value createOrFoldNot(Location loc, Value value, OpBuilder &builder);
Value createOrFoldNot(Value value, ImplicitLocOpBuilder &builder);

} // namespace chalk
} // namespace circt

#endif // CIRCT_DIALECT_CHALK_CHALKOPS_H
