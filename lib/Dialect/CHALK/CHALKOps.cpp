//===- CHALKOps.cpp - CHALK dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/CHALK/CHALKOps.h"
#include "circt/Dialect/CHALK/CHALKDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "circt/Dialect/CHALK/CHALKOps.cpp.inc"
