//===- CHALKDialect.cpp - Implement the CHALK dialect -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CHALK dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/CHALK/CHALKDialect.h"
#include "circt/Dialect/CHALK/CHALKOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace circt::chalk;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/CHALK/CHALKTypes.cpp.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/CHALK/CHALK.cpp.inc"

void CHALKDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/CHALK/CHALK.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/CHALK/CHALKTypes.cpp.inc"
      >();
}

// Provide implementations for the enums we use.
#include "circt/Dialect/CHALK/CHALKEnums.cpp.inc"

#include "circt/Dialect/CHALK/CHALKDialect.cpp.inc"

// void CXYType::print(AsmPrinter &printer) const {
//   printer << "<";
//   // Don't print element types with "!firrtl.".
//   firrtl::printNestedType(getElementType(), printer);
//   printer << ", " << getNumElements() << ">";
// }
// 
// Type CXYType::parse(AsmParser &parser) {
//   FIRRTLType elementType;
//   uint64_t numElements;
//   if (parser.parseLess() || firrtl::parseNestedType(elementType, parser) ||
//       parser.parseComma() || parser.parseInteger(numElements) ||
//       parser.parseGreater())
//     return {};
//   return parser.getChecked<CXYType>(elementType, numElements);
// }
// 
// LogicalResult CXYType::verify(function_ref<InFlightDiagnostic()> emitError,
//                                   FIRRTLType elementType,
//                                   uint64_t numElements) {
//   if (!elementType.isPassive()) {
//     return emitError() << "behavioral memory element type must be passive";
//   }
//   return success();
// }
