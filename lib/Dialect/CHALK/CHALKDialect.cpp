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
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace chalk;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void CHALKDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/CHALK/CHALK.cpp.inc"
      >();
}

/// Registered hook to materialize a single constant operation from a given
/// attribute value with the desired resultant type. This method should use
/// the provided builder to create the operation without changing the
/// insertion position. The generated operation is expected to be constant
/// like, i.e. single result, zero operands, non side-effecting, etc. On
/// success, this hook should return the value generated to represent the
/// constant value. Otherwise, it should return null on failure.
Operation *CHALKDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  // Integer constants.
  if (auto intType = type.dyn_cast<IntegerType>())
    if (auto attrValue = value.dyn_cast<IntegerAttr>())
      return builder.create<hw::ConstantOp>(loc, type, attrValue);

  // Parameter expressions materialize into hw.param.value.
  auto parentOp = builder.getBlock()->getParentOp();
  auto curModule = dyn_cast<hw::HWModuleOp>(parentOp);
  if (!curModule)
    curModule = parentOp->getParentOfType<hw::HWModuleOp>();
  if (curModule && isValidParameterExpression(value, curModule))
    return builder.create<hw::ParamValueOp>(loc, type, value);

  return nullptr;
}

// Provide implementations for the enums we use.
#include "circt/Dialect/CHALK/CHALKEnums.cpp.inc"

#include "circt/Dialect/CHALK/CHALKDialect.cpp.inc"
