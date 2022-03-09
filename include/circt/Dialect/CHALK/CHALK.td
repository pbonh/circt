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

  // let extraClassDeclaration = [{
  //   Block *getBody() { return &body().front(); }

  //   using iterator = Block::iterator;
  //   iterator begin() { return getBody()->begin(); }
  //   iterator end() { return getBody()->end(); }

  //   Block::BlockArgListType getArguments() {
  //     return getBody()->getArguments();
  //   }

  //   // Return the block argument for the port with the specified index.
  //   BlockArgument getArgument(size_t portNumber);

  //   OpBuilder getBodyBuilder() {
  //     assert(!body().empty() && "Unexpected empty 'body' region.");
  //     Block &bodyBlock = body().front();
  //     return OpBuilder::atBlockEnd(&bodyBlock);
  //   }

  //   void getAsmBlockArgumentNames(mlir::Region &region,
  //                                 mlir::OpAsmSetValueNameFn setNameFn);
  // }];

}

include "circt/Dialect/HW/HWTypes.td"

#endif // CHALK_TD
