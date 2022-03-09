//===- CHALKDialect.h - CHALK dialect declaration -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CHALK MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CHALK_CHALKDIALECT_H
#define CIRCT_DIALECT_CHALK_CHALKDIALECT_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

// Pull in the Dialect definition.
#include "circt/Dialect/CHALK/CHALKDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
#include "circt/Dialect/CHALK/CHALKEnums.h.inc"

#endif // CIRCT_DIALECT_CHALK_CHALKDIALECT_H
