//===- CHALK.td - CHALK dialect definition --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the CHALK dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CHALK_TD
#define CHALK_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def CHALKDialect : Dialect {
  let name = "chalk";

  let summary = "Types and operations for chalk dialect";
  let description = [{
    This dialect defines the `chalk` dialect, which is intended to be a generic
    representation of 2D VLSI layouts.
  }];
  let hasConstantMaterializer = 1;
  let cppNamespace = "::circt::chalk";
}

// Base class for the operation in this dialect.
class CHALKOp<string mnemonic, list<Trait> traits = []> :
    Op<CHALKDialect, mnemonic, traits>;

include "circt/Dialect/HW/HWTypes.td"

#endif // CHALK_TD
