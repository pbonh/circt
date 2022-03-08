//===- CHALK.td - CHALK dialect definition ---------------*- tablegen -*-===//
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
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "CHALKDialect.td"
include "CHALKOps.td"
include "CHALKStructure.td"

#endif // CHALK_TD
