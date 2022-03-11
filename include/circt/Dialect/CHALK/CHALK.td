//===- CHALK.td - Definition of CHALK dialect ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CHALK_TD
#define CIRCT_DIALECT_CHALK_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"

def CHALKDialect : Dialect {
  let name = "chalk";
  let summary = "Types and operations for CHALK dialect";
  let description = [{
    This dialect defines the `chalk` dialect, which is intended to represent
    finite-state machines.
  }];

  let cppNamespace = "::circt::chalk";
}

// Base class for the types in this dialect.
class CHALKType<string name> : TypeDef<CHALKDialect, name> {}

// Base class for the operations in this dialect.
class CHALKOp<string mnemonic, list<Trait> traits = []> :
    Op<CHALKDialect, mnemonic, traits>;

include "CHALKTypes.td"
include "CHALKOps.td"

#endif // CIRCT_DIALECT_CHALK_TD
