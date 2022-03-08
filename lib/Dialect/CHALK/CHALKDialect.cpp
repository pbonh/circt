//===- CHALKDialect.cpp - CHALK dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/CHALK/CHALKDialect.h"
#include "circt/Dialect/CHALK/CHALKOps.h"

using namespace circt;
using namespace circt::chalk;

//===----------------------------------------------------------------------===//
// CHALK dialect.
//===----------------------------------------------------------------------===//

void CHALKDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/CHALK/CHALK.cpp.inc"
      >();
}
