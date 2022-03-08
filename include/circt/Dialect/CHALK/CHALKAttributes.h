//===- CHALKAttributes.h - CHALK dialect attributes -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the CHALK dialect custom attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CHALK_CHALKATTRIBUTES_H
#define CIRCT_DIALECT_CHALK_CHALKATTRIBUTES_H

#include "circt/Dialect/CHALK/CHALKDialect.h"
#include "circt/Support/LLVM.h"

namespace circt {
namespace chalk {

//===----------------------------------------------------------------------===//
// PortDirections
//===----------------------------------------------------------------------===//

/// This represents the direction of a single port.
enum class Direction { In, Out };

/// Prints the Direction to the stream as either "in" or "out".
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Direction &dir);

namespace direction {

/// Return an output direction if \p isOutput is true, otherwise return an
/// input direction.
inline Direction get(bool isOutput) { return (Direction)isOutput; }

inline StringRef toString(Direction direction) {
  return direction == Direction::In ? "in" : "out";
}

/// Return a \p IntegerAttr containing the packed representation of an array
/// of directions.
IntegerAttr packAttribute(MLIRContext *context, ArrayRef<Direction> directions);

/// Turn a packed representation of port attributes into a vector that can
/// be worked with.
SmallVector<Direction> unpackAttribute(IntegerAttr directions);

} // namespace direction
} // namespace chalk
} // namespace circt

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/CHALK/CHALKAttributes.h.inc"

#endif // CIRCT_DIALECT_CHALK_CHALKATTRIBUTES_H
