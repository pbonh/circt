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
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def CHALKDialect : Dialect {
  let name = "chalk";
  let summary = "Types and operations for chalk dialect";
  let description = [{
    This dialect defines the `chalk` dialect, which is intended to be a generic
    representation of 2D VLSI layouts.
  }];
  let cppNamespace = "::circt::chalk";
}

// Base class for the operation in this dialect.
class CHALKOp<string mnemonic, list<Trait> traits = []> :
    Op<CHALKDialect, mnemonic, traits>;

def CCellOp : CHALKOp<"cell", [IsolatedFromAbove, SingleBlock,
                                    OpAsmOpInterface]> {
  let summary = "CHALK Cell";
  let description = [{
    The "chalk.cell" operation represents a CHALK Cell, similar to 
    GDSII/OASIS.
  }];

  let arguments = (ins);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name)>
  ];
}

// def CXYType : TypeDef<CHALKDialect, "CXY"> {
//   let mnemonic = "cxy";
//   let summary = "a pair of (x,y) coordinates";
//   let description = [{
//     Syntax:
//     ```
//     cxy-type ::= `cxy` `<` x-coord, y-coord `>`
//     ```
// 
//     A pair of (x,y) coordinates.
// 
//     Examples:
// 
//     ```mlir
//     !chirrtl.cxy<0, 0>
//     !chirrtl.cxy<1, 27>
//     ```
//   }];
// 
//   let parameters = (ins "uint64_t":$xCoord, "uint64_t":$yCoord);
// 
//   // let builders = [
//   //   TypeBuilderWithInferredContext<(ins "uint64_t":$xCoord,
//   //                                       "uint64_t":$yCoord), [{
//   //     return $_get(elementType.getContext(), elementType, numElements);
//   //   }]>
//   // ];
//   let genVerifyDecl = 1;
//   // let assemblyFormat = "`< x=` $x `, y=` $y `>`";
//   let assemblyFormat = "`<` $xCoord `,` $yCoord `>`";
// }

def CRectangleOp : CHALKOp<"rectangle", [IsolatedFromAbove, SingleBlock,
                                    OpAsmOpInterface]> {
  let summary = "CHALK Rectangle";
  let description = [{
    The "chalk.rectangle" operation represents a CHALK Rectangle, similar to 
    GDSII/OASIS.
  }];

  let arguments = (ins);
  // let arguments = (ins CXYType:$coord);
  let results = (outs I32:$width, I32:$height, I32:$xCoord, I32:$yCoord);
  let regions = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilder<(ins "StringAttr":$name)>
  ];
}

// include "circt/Dialect/HW/HWTypes.td"

#endif // CHALK_TD
