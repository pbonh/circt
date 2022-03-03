//===- CHALKDialect.td - CHALK dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHALK_DIALECT
#define CHALK_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// CHALK dialect definition.
//===----------------------------------------------------------------------===//

def CHALK_Dialect : Dialect {
    let name = "chalk";
    let summary = "A chalk out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::chalk";
}

//===----------------------------------------------------------------------===//
// Base chalk operation definition.
//===----------------------------------------------------------------------===//

class CHALK_Op<string mnemonic, list<Trait> traits = []> :
        Op<CHALK_Dialect, mnemonic, traits>;

#endif // CHALK_DIALECT
