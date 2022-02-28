//===- CHALKOps.td - CHALK dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CHALK_OPS
#define CHALK_OPS

include "CHALKDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def CHALK_CellOp : CHALK_Op<"cell", [NoSideEffect]> {
    let summary = "A CHALK Cell represents heirarchy in the layout.";
    let description = [{
        The `chalk.cell` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the cell operation to %0
        %1 = chalk.cell %0 : i32
        ```
    }];

    let arguments = (ins NoneType:$input);
    let results = (outs NoneType:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // CHALK_OPS
