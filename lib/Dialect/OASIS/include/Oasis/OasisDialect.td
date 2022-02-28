//===- OasisDialect.td - Oasis dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OASIS_DIALECT
#define OASIS_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Oasis dialect definition.
//===----------------------------------------------------------------------===//

def Oasis_Dialect : Dialect {
    let name = "oasis";
    let summary = "A oasis out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::oasis";
}

//===----------------------------------------------------------------------===//
// Base oasis operation definition.
//===----------------------------------------------------------------------===//

class Oasis_Op<string mnemonic, list<Trait> traits = []> :
        Op<Oasis_Dialect, mnemonic, traits>;

#endif // OASIS_DIALECT
