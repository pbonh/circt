//===- OasisDialect.cpp - Oasis dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Oasis/OasisDialect.h"
#include "Oasis/OasisOps.h"

using namespace mlir;
using namespace mlir::oasis;

//===----------------------------------------------------------------------===//
// Oasis dialect.
//===----------------------------------------------------------------------===//

void OasisDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Oasis/OasisOps.cpp.inc"
      >();
}
