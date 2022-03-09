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

#define GET_OP_CLASSES
#include "circt/Dialect/CHALK/CHALK.h.inc"

namespace circt {
namespace chalk {

} // namespace chalk
} // namespace circt

#endif // CIRCT_DIALECT_CHALK_CHALKOPS_H
