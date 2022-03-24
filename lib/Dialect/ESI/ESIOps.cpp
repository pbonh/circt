//===- ESIOps.cpp - ESI op code defs ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is where op definitions live.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

using namespace circt;
using namespace circt::esi;

//===----------------------------------------------------------------------===//
// ChannelBuffer functions.
//===----------------------------------------------------------------------===//

ParseResult ChannelBuffer::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();

  llvm::SmallVector<OpAsmParser::OperandType, 4> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/3,
                              /*delimiter=*/OpAsmParser::Delimiter::None))
    return failure();

  ChannelBufferOptions optionsAttr;
  if (parser.parseAttribute(optionsAttr,
                            parser.getBuilder().getType<NoneType>(), "options",
                            result.attributes))
    return failure();

  Type innerOutputType;
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(innerOutputType))
    return failure();
  auto outputType =
      ChannelPort::get(parser.getBuilder().getContext(), innerOutputType);
  result.addTypes({outputType});

  auto i1 = IntegerType::get(result.getContext(), 1);
  if (parser.resolveOperands(operands, {i1, i1, outputType}, inputOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

void ChannelBuffer::print(OpAsmPrinter &p) {
  p << " " << clk() << ", " << rstn() << ", " << input() << " ";
  p.printAttributeWithoutType(options());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"options"});
  p << " : " << output().getType().cast<ChannelPort>().getInner();
}

//===----------------------------------------------------------------------===//
// PipelineStage functions.
//===----------------------------------------------------------------------===//

ParseResult PipelineStage::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();

  SmallVector<OpAsmParser::OperandType, 4> operands;
  Type innerOutputType;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/3) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(innerOutputType))
    return failure();
  auto type =
      ChannelPort::get(parser.getBuilder().getContext(), innerOutputType);
  result.addTypes({type});

  auto i1 = IntegerType::get(result.getContext(), 1);
  if (parser.resolveOperands(operands, {i1, i1, type}, inputOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

void PipelineStage::print(OpAsmPrinter &p) {
  p << " " << clk() << ", " << rstn() << ", " << input() << " ";
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << output().getType().cast<ChannelPort>().getInner();
}

//===----------------------------------------------------------------------===//
// Wrap / unwrap.
//===----------------------------------------------------------------------===//

ParseResult WrapValidReady::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();

  llvm::SmallVector<OpAsmParser::OperandType, 2> opList;
  Type innerOutputType;
  if (parser.parseOperandList(opList, 2, OpAsmParser::Delimiter::None) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(innerOutputType))
    return failure();

  auto boolType = parser.getBuilder().getI1Type();
  auto outputType =
      ChannelPort::get(parser.getBuilder().getContext(), innerOutputType);
  result.addTypes({outputType, boolType});
  if (parser.resolveOperands(opList, {innerOutputType, boolType},
                             inputOperandsLoc, result.operands))
    return failure();
  return success();
}

void WrapValidReady::print(OpAsmPrinter &p) {
  p << " " << rawInput() << ", " << valid();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << chanOutput().getType().cast<ChannelPort>().getInner();
}

void WrapValidReady::build(OpBuilder &b, OperationState &state, Value data,
                           Value valid) {
  build(b, state, ChannelPort::get(state.getContext(), data.getType()),
        b.getI1Type(), data, valid);
}

ParseResult UnwrapValidReady::parse(OpAsmParser &parser,
                                    OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();

  llvm::SmallVector<OpAsmParser::OperandType, 2> opList;
  Type outputType;
  if (parser.parseOperandList(opList, 2, OpAsmParser::Delimiter::None) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(outputType))
    return failure();

  auto inputType =
      ChannelPort::get(parser.getBuilder().getContext(), outputType);

  auto boolType = parser.getBuilder().getI1Type();

  result.addTypes({inputType.getInner(), boolType});
  if (parser.resolveOperands(opList, {inputType, boolType}, inputOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

void UnwrapValidReady::print(OpAsmPrinter &p) {
  p << " " << chanInput() << ", " << ready();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << rawOutput().getType();
}

void UnwrapValidReady::build(OpBuilder &b, OperationState &state, Value inChan,
                             Value ready) {
  auto inChanType = inChan.getType().cast<ChannelPort>();
  build(b, state, inChanType.getInner(), b.getI1Type(), inChan, ready);
}

/// If 'iface' looks like an ESI interface, return the inner data type.
static Type getEsiDataType(circt::sv::InterfaceOp iface) {
  using namespace circt::sv;
  if (!iface.lookupSymbol<InterfaceSignalOp>("valid"))
    return Type();
  if (!iface.lookupSymbol<InterfaceSignalOp>("ready"))
    return Type();
  auto dataSig = iface.lookupSymbol<InterfaceSignalOp>("data");
  if (!dataSig)
    return Type();
  return dataSig.type();
}

/// Verify that the modport type of 'modportArg' points to an interface which
/// looks like an ESI interface and the inner data from said interface matches
/// the chan type's inner data type.
static LogicalResult verifySVInterface(Operation *op,
                                       circt::sv::ModportType modportType,
                                       ChannelPort chanType) {
  auto modport =
      SymbolTable::lookupNearestSymbolFrom<circt::sv::InterfaceModportOp>(
          op, modportType.getModport());
  if (!modport)
    return op->emitError("Could not find modport ")
           << modportType.getModport() << " in symbol table.";
  auto iface = cast<circt::sv::InterfaceOp>(modport->getParentOp());
  Type esiDataType = getEsiDataType(iface);
  if (!esiDataType)
    return op->emitOpError("Interface is not a valid ESI interface.");
  if (esiDataType != chanType.getInner())
    return op->emitOpError("Operation specifies ")
           << chanType << " but type inside doesn't match interface data type "
           << esiDataType << ".";
  return success();
}

LogicalResult WrapSVInterface::verify() {
  auto modportType = interfaceSink().getType().cast<circt::sv::ModportType>();
  auto chanType = output().getType().cast<ChannelPort>();
  return verifySVInterface(*this, modportType, chanType);
}

LogicalResult UnwrapSVInterface::verify() {
  auto modportType = interfaceSource().getType().cast<circt::sv::ModportType>();
  auto chanType = chanInput().getType().cast<ChannelPort>();
  return verifySVInterface(*this, modportType, chanType);
}

#define GET_OP_CLASSES
#include "circt/Dialect/ESI/ESI.cpp.inc"
