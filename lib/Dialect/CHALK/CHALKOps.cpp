//===- CHALKOps.cpp - Implementation of CHALK dialect operations --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/CHALK/CHALKOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace circt;
using namespace chalk;

//===----------------------------------------------------------------------===//
// CellOp
//===----------------------------------------------------------------------===//

void CellOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                      Type stateType, FunctionType type,
                      ArrayRef<NamedAttribute> attrs,
                      ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute("stateType", TypeAttr::get(stateType));
  state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(builder, state, argAttrs,
                                                /*resultAttrs=*/llvm::None);
}

/// Get the port information of the machine.
void CellOp::getHWPortInfo(SmallVectorImpl<hw::PortInfo> &ports) {
  ports.clear();
  auto machineType = getType();
  auto builder = Builder(*this);

  for (unsigned i = 0, e = machineType.getNumInputs(); i < e; ++i) {
    hw::PortInfo port;
    port.name = builder.getStringAttr("in" + std::to_string(i));
    port.direction = circt::hw::PortDirection::INPUT;
    port.type = machineType.getInput(i);
    port.argNum = i;
    ports.push_back(port);
  }

  for (unsigned i = 0, e = machineType.getNumResults(); i < e; ++i) {
    hw::PortInfo port;
    port.name = builder.getStringAttr("out" + std::to_string(i));
    port.direction = circt::hw::PortDirection::OUTPUT;
    port.type = machineType.getResult(i);
    port.argNum = i;
    ports.push_back(port);
  }
}

ParseResult CellOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false, buildFuncType);
}

void CellOp::print(OpAsmPrinter &p) {
  FunctionType fnType = getType();
  function_interface_impl::printFunctionOp(
      p, *this, fnType.getInputs(), /*isVariadic=*/false, fnType.getResults());
}

static LogicalResult compareTypes(TypeRange rangeA, TypeRange rangeB) {
  if (rangeA.size() != rangeB.size())
    return failure();

  int64_t index = 0;
  for (auto zip : llvm::zip(rangeA, rangeB)) {
    if (std::get<0>(zip) != std::get<1>(zip))
      return failure();
    ++index;
  }

  return success();
}

static LogicalResult verifyCellOp(CellOp op) {
  // If this function is external there is nothing to do.
  if (op.isExternal())
    return success();

  if (!op.stateType().isa<IntegerType>())
    return op.emitOpError("state must be integer type");

  // Verify that the argument list of the function and the arg list of the entry
  // block line up.  The trait already verified that the number of arguments is
  // the same between the signature and the block.
  if (failed(compareTypes(op.getType().getInputs(),
                          op.front().getArgumentTypes())))
    return op.emitOpError(
        "entry block argument types must match the machine input types");

  // Verify that the machine only has one block terminated with OutputOp.
  if (!llvm::hasSingleElement(op))
    return op.emitOpError("must only have a single block");

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/CHALK/CHALK.cpp.inc"
#undef GET_OP_CLASSES

#include "circt/Dialect/CHALK/CHALKDialect.cpp.inc"
