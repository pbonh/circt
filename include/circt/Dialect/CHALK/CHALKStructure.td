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

#ifndef CHALK_STRUCTURE
#define CHALK_STRUCTURE

include "CHALKDialect.td"

def ParamDeclAttr : AttrDef<CHALKDialect, "ParamDecl"> {
  let summary = "module or instance parameter definition";

  /// The value of the attribute - in a module, this is the default
  /// value (and may be missing).  In an instance, this is a required field that
  /// specifies the value being passed.  The verilog emitter omits printing the
  /// parameter for an instance when the applied value and the default value are
  /// the same.
  let parameters = (ins "::mlir::StringAttr":$name,
                        "::mlir::TypeAttr":$type,
                        "::mlir::Attribute":$value);
  let mnemonic = "param.decl";

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$name,
                                         "::mlir::Type":$type),
      "auto *context = type.getContext();\n"
      "return $_get(context, name, TypeAttr::get(type), Attribute());">,
    AttrBuilderWithInferredContext<(ins "::mlir::StringRef":$name,
                                         "::mlir::Type":$type),
      "return get(StringAttr::get(type.getContext(), name), type);">,

    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$name,
                                        "::mlir::Attribute":$value),
      "auto *context = value.getContext();\n"
      "return $_get(context, name, TypeAttr::get(value.getType()), value);">,
    AttrBuilderWithInferredContext<(ins "::mlir::StringRef":$name,
                                        "::mlir::Attribute":$value),
      "return get(StringAttr::get(value.getContext(), name), value);">
  ];

  let extraClassDeclaration = [{
    static ParamDeclAttr getWithName(ParamDeclAttr param,
                                     ::mlir::StringAttr name) {
      return get(param.getContext(), name, param.getType(), param.getValue());
    }
  }];
}

def ParamDeclArrayAttr
  : TypedArrayAttrBase<ParamDeclAttr, "parameter array attribute">;

def FModuleLike : OpInterface<"FModuleLike"> {
  let cppNamespace = "circt::firrtl";
  let description = "Provide common  module information.";
  let methods = [

    //===------------------------------------------------------------------===//
    // Module Name
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the module name",
    "StringRef", "moduleName", (ins),
    /*methodBody=*/[{ return $_op.getName(); }]>,

    InterfaceMethod<"Get the module name",
    "StringAttr", "moduleNameAttr", (ins),
    /*methodBody=*/[{ return $_op.getNameAttr(); }]>,

    //===------------------------------------------------------------------===//
    // Port Directions
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the port directions attribute",
    "IntegerAttr", "getPortDirectionsAttr", (ins), [{}],
    /*defaultImplementation=*/[{
      return $_op->template
        getAttrOfType<IntegerAttr>(FModuleLike::getPortDirectionsAttrName());
    }]>,

    InterfaceMethod<"Get the port directions",
    "APInt", "getPortDirections", (ins), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortDirectionsAttr().getValue();
    }]>,

    InterfaceMethod<"Get a port direction",
    "Direction", "getPortDirection", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      return direction::get($_op.getPortDirections()[portIndex]);
    }]>,

    //===------------------------------------------------------------------===//
    // Port Names
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the port names attribute",
    "ArrayAttr", "getPortNamesAttr", (ins), [{}],
    /*defaultImplementation=*/[{
      return $_op->template
        getAttrOfType<ArrayAttr>(FModuleLike::getPortNamesAttrName());
    }]>,

    InterfaceMethod<"Get the port names",
    "ArrayRef<Attribute>", "getPortNames", (ins), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortNamesAttr().getValue();
    }]>,

    InterfaceMethod<"Get a port name",
    "StringAttr", "getPortNameAttr", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortNames()[portIndex].template cast<StringAttr>();
    }]>,

    InterfaceMethod<"Get a port name",
    "StringRef", "getPortName", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortNameAttr(portIndex).getValue();
    }]>,

    //===------------------------------------------------------------------===//
    // Port Types
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the port types attribute",
    "ArrayAttr", "getPortTypesAttr", (ins), [{}],
    /*defaultImplementation=*/[{
      return $_op->template
        getAttrOfType<ArrayAttr>(FModuleLike::getPortTypesAttrName());
    }]>,

    InterfaceMethod<"Get the port types",
    "ArrayRef<Attribute>", "getPortTypes", (ins), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortTypesAttr().getValue();
    }]>,

    InterfaceMethod<"Get a port type",
    "TypeAttr", "getPortTypeAttr", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortTypes()[portIndex].template cast<TypeAttr>();
    }]>,

    InterfaceMethod<"Get a port type",
    "FIRRTLType", "getPortType", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      auto typeAttr = $_op.getPortTypeAttr(portIndex);
      return typeAttr.getValue().template cast<FIRRTLType>();
    }]>,

    //===------------------------------------------------------------------===//
    // Port Annotations
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the port annotations attribute",
    "ArrayAttr", "getPortAnnotationsAttr", (ins), [{}],
    /*defaultImplementation=*/[{
      return $_op->template
        getAttrOfType<ArrayAttr>(FModuleLike::getPortAnnotationsAttrName());
    }]>,

    InterfaceMethod<"Get the port annotations attribute",
    "ArrayRef<Attribute>", "getPortAnnotations", (ins), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortAnnotationsAttr().getValue();
    }]>,

    InterfaceMethod<"Get a port's annotations attribute",
    "ArrayAttr", "getAnnotationsAttrForPort", (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      auto annos = $_op.getPortAnnotationsAttr();
      if (annos.empty())
        return annos; // annos already is an empty array
      return annos[portIndex].template cast<ArrayAttr>();
    }]>,

    InterfaceMethod<"Get a port's annotations",
    "ArrayRef<Attribute>", "getAnnotationsForPort",
    (ins "size_t":$portIndex), [{}],
    /*defaultImplementation=*/[{
      return $_op.getAnnotationsAttrForPort(portIndex).getValue();
    }]>,

    //===------------------------------------------------------------------===//
    // Port Symbols
    //===------------------------------------------------------------------===//

    // Getters
    InterfaceMethod<"Get the port symbols attribute",
    "ArrayAttr", "getPortSymbolsAttr", (ins), [{}], [{
      return $_op->template
        getAttrOfType<ArrayAttr>(FModuleLike::getPortSymbolsAttrName());
    }]>,

    InterfaceMethod<"Get the port symbols",
    "ArrayRef<Attribute>", "getPortSymbols", (ins), [{}], [{
      return $_op.getPortSymbolsAttr().getValue();
    }]>,

    InterfaceMethod<"Get a port symbol attribute",
    "StringAttr", "getPortSymbolAttr", (ins "size_t":$portIndex), [{}], [{
      auto syms = $_op.getPortSymbols();
      if (syms.empty())
        return StringAttr::get($_op.getContext(), "");
      return syms[portIndex].template cast<StringAttr>();
    }]>,

    InterfaceMethod<"Get a port symbol",
    "StringRef", "getPortSymbol", (ins "size_t":$portIndex), [{}], [{
      return $_op.getPortSymbolAttr(portIndex).getValue();
    }]>,

    // Setters
    InterfaceMethod<"Set the port symbols attribute", "void",
    "setPortSymbolsAttr", (ins "ArrayAttr":$symbols), [{}], [{
      if (llvm::all_of(symbols.getValue(), [](Attribute symbol) {
         return symbol.cast<StringAttr>().getValue().empty(); }))
        symbols = ArrayAttr::get($_op.getContext(), {});
      assert(symbols.getValue().empty() ||
             symbols.getValue().size() == $_op.getNumPorts());
      $_op->setAttr(FModuleLike::getPortSymbolsAttrName(), symbols);
    }]>,

    InterfaceMethod<"Set the port symbols", "void",
    "setPortSymbols", (ins "ArrayRef<Attribute>":$symbols), [{}], [{
      if (llvm::all_of(symbols, [](Attribute symbol) {
         return symbol.cast<StringAttr>().getValue().empty(); }))
        symbols = {};
      assert(symbols.empty() || symbols.size() == $_op.getNumPorts());
      $_op.setPortSymbolsAttr(ArrayAttr::get($_op.getContext(), symbols));
    }]>,

    InterfaceMethod<"Set a port symbol attribute", "void",
    "setPortSymbolAttr", (ins "size_t":$portIndex, "StringAttr":$symbol), [{}],
    [{
      SmallVector<Attribute> symbols($_op.getPortSymbols().begin(),
                                     $_op.getPortSymbols().end());
      if (symbols.empty()) {
        auto emptyString = StringAttr::get($_op.getContext(), "");
        symbols.resize($_op.getNumPorts(), emptyString);
      }
      assert(symbols.size() == $_op.getNumPorts());
      symbols[portIndex] = symbol;
      $_op.setPortSymbols(symbols);
    }]>,

    InterfaceMethod<"Set a port symbol", "void",
    "setPortSymbol", (ins "size_t":$portIndex, "StringRef":$symbol), [{}], [{
      $_op.setPortSymbolAttr(portIndex, StringAttr::get($_op.getContext(),
        symbol));
    }]>,

    //===------------------------------------------------------------------===//
    // All Port Information
    //===------------------------------------------------------------------===//

    InterfaceMethod<"Get the number of ports",
    "size_t", "getNumPorts", (ins), [{}],
    /*defaultImplementation=*/[{
      return $_op.getPortTypesAttr().size();
    }]>,

    InterfaceMethod<"Get information about all ports",
    "SmallVector<PortInfo>", "getPorts">
  ];

  let extraClassDeclaration = [{
    /// Get the attribute name for port directions.
    static StringRef getPortDirectionsAttrName() {
      return "portDirections";
    }

    /// Get the attribute name for port names",
    static StringRef getPortNamesAttrName() {
      return "portNames";
    }

    /// Get the attribute name for port types.
    static StringRef getPortTypesAttrName() {
      return "portTypes";
    }

    /// Get the attribute name for port annotations.
    static StringRef getPortAnnotationsAttrName() {
      return "portAnnotations";
    }

    /// Get the attribute name for port symbols.
    static StringRef getPortSymbolsAttrName() {
      return "portSyms";
    }
  }];

  let verify = [{
    static_assert(
        ConcreteOp::template hasTrait<::mlir::SymbolOpInterface::Trait>(),
        "expected operation to be a symbol");
    return verifyModuleLikeOpInterface(op);
  }];
}

def CircuitOp : CHALK_Op<"circuit",
      [IsolatedFromAbove, SymbolTable, SingleBlock, NoTerminator,
       NoRegionArguments]> {
  let summary = "CHALK Circuit";
  let description = [{
    The "chalk.circuit" operation represents an overall Verilog circuit,
    containing a list of modules.
  }];
  let arguments = (ins StrAttr:$name);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name)>
  ];

  let extraClassDeclaration = [{
    OpBuilder getBodyBuilder() {
      assert(!body().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = body().front();
      return OpBuilder::atBlockEnd(&bodyBlock);
    }

    /// Return body of this circuit.
    Region &getBodyRegion();
    Block *getBody();

    // Return the main module that is the entry point of the circuit.  This may
    // be either an FModuleOp or an FExtModuleOp.
    Operation *getMainModule();
  }];

  let assemblyFormat = "$name custom<CircuitOpAttrs>(attr-dict) $body";
  let verifier = "return ::verifyCircuitOp(*this);";
}

def FModuleOp : CHALK_Op<"module", [IsolatedFromAbove, Symbol, SingleBlock,
                                    NoTerminator, HasParent<"CircuitOp">,
                                    DeclareOpInterfaceMethods<FModuleLike>,
                                    OpAsmOpInterface]> {
  let summary = "CHALK Module";
  let description = [{
    The "chalk.module" operation represents a Verilog module, including a given
    name, a list of ports, and a body that represents the connections within
    the module.
  }];
  let arguments = (ins StrAttr:$name);

  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<PortInfo>":$ports)>
  ];

  let extraClassDeclaration = [{
    Block *getBody() { return &body().front(); }

    using iterator = Block::iterator;
    iterator begin() { return getBody()->begin(); }
    iterator end() { return getBody()->end(); }

    Block::BlockArgListType getArguments() {
      return getBody()->getArguments();
    }

    // Return the block argument for the port with the specified index.
    BlockArgument getArgument(size_t portNumber);

    OpBuilder getBodyBuilder() {
      assert(!body().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = body().front();
      return OpBuilder::atBlockEnd(&bodyBlock);
    }

    /// Inserts the given ports.
    void insertPorts(ArrayRef<std::pair<unsigned, PortInfo>> ports);

    /// Erases the ports listed in `portIndices`.  `portIndices` is expected to
    /// be in order and unique.
    void erasePorts(ArrayRef<unsigned> portIndices);

    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);
  }];

}

def FExtModuleOp : CHALK_Op<"extmodule",
      [IsolatedFromAbove, Symbol, HasParent<"CircuitOp">,
       DeclareOpInterfaceMethods<FModuleLike>, OpAsmOpInterface]> {
  let summary = "CHALK extmodule";
  let description = [{
    The "chalk.extmodule" operation represents an external reference to a
    Verilog module, including a given name and a list of ports.
  }];
  let arguments = (ins
                   OptionalAttr<StrAttr>:$defname,
                   ParamDeclArrayAttr:$parameters
                  );
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name,
                      "ArrayRef<PortInfo>":$ports,
                      CArg<"StringRef", "StringRef()">:$defnamAttr,
                      CArg<"ArrayAttr", "ArrayAttr()">:$parameters)>
  ];

  let extraClassDeclaration = [{
    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);
  }];

  let hasCustomAssemblyFormat = 1;
  let verifier = "return ::verifyFExtModuleOp(*this);";
}

#endif // CHALK_STRUCTURE
