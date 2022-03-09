//===- CHALKVisitors.h - CHALK Dialect Visitors -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines visitors that make it easier to work with the CHALK IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CHALK_CHALKVISITORS_H
#define CIRCT_DIALECT_CHALK_CHALKVISITORS_H

#include "circt/Dialect/CHALK/CHALKOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace chalk {

/// This helps visit CHALK nodes.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class CHALKVisitor {
public:
  ResultType dispatchCHALKVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<
            // Arithmetic and Logical Binary Operations.
            AddOp, SubOp, MulOp, DivUOp, DivSOp, ModUOp, ModSOp, ShlOp, ShrUOp,
            ShrSOp,
            // Bitwise operations
            AndOp, OrOp, XorOp,
            // Comparison operations
            ICmpOp,
            // Reduction Operators
            ParityOp,
            // Other operations.
            ConcatOp, ReplicateOp, ExtractOp, MuxOp>(
            [&](auto expr) -> ResultType {
              return thisCast->visitCHALK(expr, args...);
            })
        .Default([&](auto expr) -> ResultType {
          return thisCast->visitInvalidCHALK(op, args...);
        });
  }

  /// This callback is invoked on any non-expression operations.
  ResultType visitInvalidCHALK(Operation *op, ExtraArgs... args) {
    op->emitOpError("unknown chalk node");
    abort();
  }

  /// This callback is invoked on any chalk operations that are not
  /// handled by the concrete visitor.
  ResultType visitUnhandledCHALK(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

  /// This fallback is invoked on any binary node that isn't explicitly handled.
  /// The default implementation delegates to the 'unhandled' fallback.
  ResultType visitBinaryCHALK(Operation *op, ExtraArgs... args) {
    return static_cast<ConcreteType *>(this)->visitUnhandledCHALK(op, args...);
  }

  ResultType visitUnaryCHALK(Operation *op, ExtraArgs... args) {
    return static_cast<ConcreteType *>(this)->visitUnhandledCHALK(op, args...);
  }

  ResultType visitVariadicCHALK(Operation *op, ExtraArgs... args) {
    return static_cast<ConcreteType *>(this)->visitUnhandledCHALK(op, args...);
  }

#define HANDLE(OPTYPE, OPKIND)                                                 \
  ResultType visitCHALK(OPTYPE op, ExtraArgs... args) {                         \
    return static_cast<ConcreteType *>(this)->visit##OPKIND##CHALK(op,          \
                                                                  args...);    \
  }

  // Arithmetic and Logical Binary Operations.
  HANDLE(AddOp, Binary);
  HANDLE(SubOp, Binary);
  HANDLE(MulOp, Binary);
  HANDLE(DivUOp, Binary);
  HANDLE(DivSOp, Binary);
  HANDLE(ModUOp, Binary);
  HANDLE(ModSOp, Binary);
  HANDLE(ShlOp, Binary);
  HANDLE(ShrUOp, Binary);
  HANDLE(ShrSOp, Binary);

  HANDLE(AndOp, Variadic);
  HANDLE(OrOp, Variadic);
  HANDLE(XorOp, Variadic);

  HANDLE(ParityOp, Unary);

  HANDLE(ICmpOp, Binary);

  // Other operations.
  HANDLE(ConcatOp, Unhandled);
  HANDLE(ReplicateOp, Unhandled);
  HANDLE(ExtractOp, Unhandled);
  HANDLE(MuxOp, Unhandled);
#undef HANDLE
};

} // namespace chalk
} // namespace circt

#endif // CIRCT_DIALECT_CHALK_CHALKVISITORS_H
