//===- CHALKDialect.h - CHALK dialect declaration ------------*- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an MLIR dialect for the CHALK IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CHALK_DIALECT_H
#define CIRCT_DIALECT_CHALK_DIALECT_H

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

// Pull in the dialect definition.
#include "circt/Dialect/CHALK/CHALKDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
#include "circt/Dialect/CHALK/CHALKEnums.h.inc"

namespace circt {
class FieldRef;

namespace chalk {

/// Get the FieldRef from a value.  This will travel backwards to through the
/// IR, following Subfield and Subindex to find the op which declares the
/// location.
FieldRef getFieldRefFromValue(Value value);

/// Get a string identifier representing the FieldRef.
std::string getFieldName(const FieldRef &fieldRef);
std::string getFieldName(const FieldRef &fieldRef, bool &rootKnown);

} // namespace chalk
} // namespace circt

#endif // CIRCT_DIALECT_CHALK_IR_DIALECT_H
