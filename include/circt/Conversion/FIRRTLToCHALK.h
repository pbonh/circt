//===- FIRRTLToCHALK.h - FIRRTL to CHALK pass entry point -----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the FIRRTLToCHALK pass
// constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_FIRRTLTOCHALK_H
#define CIRCT_CONVERSION_FIRRTLTOCHALK_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {

std::unique_ptr<mlir::Pass> createConvertFIRRTLToCHALKEmbedPass();
std::unique_ptr<mlir::Pass> createConvertFIRRTLToCHALKParallelPass();

} // namespace circt

#endif // CIRCT_CONVERSION_FIRRTLTOCHALK_H
