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

def CHALKDialect : Dialect {
    let name = "chalk";
    let summary = "A 2D IC layout MLIR dialect.";
    let description = [{
        CHALK is a MLIR dialect designed to represent 2D VLSI layouts. It is based
        on GDSII/OASIS/LEF/DEF and is intended to serve as a FIRRTL target and
        a co-optimization IR.
    }];
    let cppNamespace = "::mlir::chalk";
}

//===----------------------------------------------------------------------===//
// Base chalk operation definition.
//===----------------------------------------------------------------------===//

class CHALK_Op<string mnemonic, list<Trait> traits = []> :
        Op<CHALKDialect, mnemonic, traits>;

#endif // CHALK_DIALECT
