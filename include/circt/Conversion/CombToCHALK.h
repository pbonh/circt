//===- CombToCHALK.h - Comb to CHALK pass entry point -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the CombToCHALK pass
// constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_COMBTOCHALK_H
#define CIRCT_CONVERSION_COMBTOCHALK_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {

/// Create a Comb to CHALK conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> createConvertCombToCHALKPass();

} // namespace circt

#endif // CIRCT_CONVERSION_COMBTOCHALK_H
