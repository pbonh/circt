//===- OASIS.td - OASIS dialect definition ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the OASIS dialect.
//
//===----------------------------------------------------------------------===//

#ifndef OASIS_TD
#define OASIS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "OASISOps.td"

#endif // OASIS_TD
