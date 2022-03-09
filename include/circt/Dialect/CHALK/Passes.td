//===-- Passes.td - CHALK pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the CHALK dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CHALK_PASSES_TD
#define CIRCT_DIALECT_CHALK_PASSES_TD

include "mlir/Pass/PassBase.td"

def CHALKInitialPlacement : Pass<"chalk-init-placement", "chalk::CCellOp"> {
  let summary = "place CHALK cells";
  let description = [{
    Provide an initial placement for CHALK cells with no overlap.
  }];
  let constructor = "circt::chalk::createCHALKInitialPlacement()";
  let options = [];
  let dependentDialects = ["hw::HWDialect"];
}

#endif // CIRCT_DIALECT_CHALK_PASSES_TD
