//===- OASISDialect.cpp - OASIS dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OASIS/OASISDialect.h"
#include "circt/Dialect/OASIS/OASISOps.h"

using namespace mlir;
using namespace mlir::oasis;

//===----------------------------------------------------------------------===//
// OASIS dialect.
//===----------------------------------------------------------------------===//

void OASISDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/OASIS/OASISOps.h.inc"
      >();
}
