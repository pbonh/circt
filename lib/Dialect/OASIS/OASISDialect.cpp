//===- OASISDialect.cpp - OASIS dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OASIS/OASISDialect.h"
#include "OASIS/OASISOps.h"

using namespace mlir;
using namespace mlir::oasis;

//===----------------------------------------------------------------------===//
// OASIS dialect.
//===----------------------------------------------------------------------===//

void OASISDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "OASIS/OASISOps.cpp.inc"
      >();
}
