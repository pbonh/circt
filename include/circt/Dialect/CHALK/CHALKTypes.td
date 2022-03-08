//===- CHALKTypes.td - CHALK Types Definition ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the CHALK types.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CHALK Types Definitions
//===----------------------------------------------------------------------===//

def CHALKType : DialectType<CHALKDialect, CPred<"$_self.isa<CHALKType>()">,
  "CHALKType", "::circt::chalk::CHALKType">;

def ClockType : DialectType<CHALKDialect, CPred<"$_self.isa<ClockType>()">,
    "clock", "::circt::chalk::ClockType">,
  BuildableType<"ClockType::get($_builder.getContext())">;

def IntType : DialectType<CHALKDialect, CPred<"$_self.isa<IntType>()">,
 "sint or uint type", "::circt::chalk::IntType">;

def SIntType : DialectType<CHALKDialect, CPred<"$_self.isa<SIntType>()">,
 "sint type", "::circt::chalk::SIntType">;

def UIntType : DialectType<CHALKDialect, CPred<"$_self.isa<UIntType>()">,
 "uint type", "::circt::chalk::UIntType">;

def AnalogType : DialectType<CHALKDialect, CPred<"$_self.isa<AnalogType>()">,
 "analog type", "::circt::chalk::AnalogType">;

def BundleType : DialectType<CHALKDialect, CPred<"$_self.isa<BundleType>()">,
 "BundleType", "::circt::chalk::BundleType">;

def FVectorType : DialectType<CHALKDialect, CPred<"$_self.isa<FVectorType>()">,
  "FVectorType", "::circt::chalk::FVectorType">;

def UInt1Type : DialectType<CHALKDialect,
    CPred<"$_self.isa<UIntType>() && "
          "($_self.cast<UIntType>().getWidth() == 1 ||"
          " $_self.cast<UIntType>().getWidth() == None)">,
    "UInt<1> or UInt", "::circt::chalk::UIntType">,
  BuildableType<"UIntType::get($_builder.getContext(), 1)">;

def AsyncResetType : DialectType<CHALKDialect,
    CPred<"$_self.isa<AsyncResetType>()">,
    "AsyncReset", "::circt::chalk::AsyncResetType">,
  BuildableType<"AsyncResetType::get($_builder.getContext())">;

def ResetType : DialectType<CHALKDialect,
    CPred<"$_self.isa<ResetType>()">,
    "Reset", "::circt::chalk::ResetType">,
  BuildableType<"ResetType::get($_builder.getContext())">;

def PassiveType : DialectType<CHALKDialect,
  CPred<"$_self.isa<CHALKType>() && $_self.cast<CHALKType>().isPassive()">,
  "a passive type (contain no flips)", "::circt::chalk::CHALKType">;

//===----------------------------------------------------------------------===//
// CHALK Types Predicates
//===----------------------------------------------------------------------===//

def OneBitType : DialectType<CHALKDialect,
 CPred<"($_self.isa<IntType>() && $_self.cast<IntType>().getWidth() == 1) || "
   "($_self.isa<AnalogType>() && $_self.cast<AnalogType>().getWidth() == 1)">,
 "UInt<1>, SInt<1>, or Analog<1>", "::circt::chalk::CHALKType">;

def AnyResetType : DialectType<CHALKDialect,
    CPred<"$_self.isa<CHALKType>() && $_self.cast<CHALKType>().isResetType()">,
    "Reset", "::circt::chalk::CHALKType">;

def UIntSIntClockType : AnyTypeOf<[SIntType, UIntType, ClockType],
                                  "sint, uint, or clock",
                                  "::circt::chalk::CHALKType">;

def OneBitCastableType : AnyTypeOf<
  [OneBitType, AnyResetType, AsyncResetType, ClockType],
  "1-bit uint/sint/analog, reset, asyncreset, or clock",
                                  "::circt::chalk::CHALKType">;

//===----------------------------------------------------------------------===//
// CHALK Enum Definitions
//===----------------------------------------------------------------------===//

/// RUWAttr: Undefined/Old/New.
def RUW_Undefined: I32EnumAttrCase<"Undefined", 0>;
def RUW_Old: I32EnumAttrCase<"Old", 1>;
def RUW_New: I32EnumAttrCase<"New", 2>;

def RUWAttr: I32EnumAttr<"RUWAttr", "Read Under Write Enum",
                         [RUW_Undefined, RUW_Old, RUW_New]>;

def MemDir_Infer    : I32EnumAttrCase<"Infer", 0>;
def MemDir_Read     : I32EnumAttrCase<"Read", 1>;
def MemDir_Write    : I32EnumAttrCase<"Write", 2>;
def MemDir_ReadWrite: I32EnumAttrCase<"ReadWrite", 3>;

def MemDirAttr: I32EnumAttr<"MemDirAttr", "Memory Direction Enum",
                            [MemDir_Infer, MemDir_Read, MemDir_Write,
                              MemDir_ReadWrite]>;
