//===- CHALKOps.td - Definition of CHALK dialect operations -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CHALK_CHALKOPS_TD
#define CIRCT_DIALECT_CHALK_CHALKOPS_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def CellOp : CHALKOp<"cell", [HasParent<"mlir::ModuleOp">, FunctionOpInterface,
      Symbol, SymbolTable, IsolatedFromAbove, NoTerminator]> {
  let summary = "Define a 2D VLSI layout";
  let description = [{
    `chalk.cell` represents a 2D VLSI layout, where the region nests
    multiple shapes, paths, or placements.
  }];

  let arguments = (ins StrAttr:$cell_name);
  // let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    /// Returns the type of this function.
    FunctionType getType() { 
      return getTypeAttr().getValue().cast<FunctionType>();
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getType().getResults(); }

    /// Verify the type attribute of this function. Returns failure and emits
    /// an error if the attribute is invalid.
    LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let verifier = [{ return ::verify$cppClass(*this); }];
}

def RectangleOp : CHALKOp<"rectangle", [HasParent<"CellOp">]> {
  let summary = "Define a state of a machine";
  let description = [{
    `chalk.rectangle` represents a rectangle in a 2D VLSI layout. It
    uses attributes to store the (x,y) coordinates and the height
    and width.
  }];
  let arguments = (ins
    I64Attr:$xCoord,
    I64Attr:$yCoord,
    UI64Attr:$width,
    UI64Attr:$height
  );
  let results = (outs);
}

#endif // CIRCT_DIALECT_CHALK_CHALKOPS_TD
