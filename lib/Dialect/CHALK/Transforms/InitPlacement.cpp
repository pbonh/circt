//===- InitPlacement.cpp - Initial Placement --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Initial Placement pass.  This pass processes CHALK
// cells, and produces a non-overlapping initial placement.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/CHALK/CHALKDialect.h"
#include "circt/Dialect/CHALK/CHALKOps.h"
#include "circt/Dialect/CHALK/CHALKTypes.h"
#include "circt/Dialect/CHALK/Passes.h"
#include "mlir/IR/Threading.h"
#include "llvm/Support/Casting.h"

using namespace circt;
using namespace chalk;

using RectangleList = std::vector< RectangleOp* >;

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct InitPlacementPass : public InitPlacementBase<InitPlacementPass> {
    void runOnOperation() override;
};
} // end anonymous namespace

void InitPlacementPass::runOnOperation() {
    auto cell = getOperation();
    RectangleList rectangles;
    llvm::for_each(cell.getBody()->getOperations(), [&](Operation &op) {
        if (isa<RectangleOp>(op)) {
            auto rectangle = dyn_cast<RectangleOp>(op);
            if (rectangle) {
                rectangles.push_back(&rectangle);
            }
        }
    });

    size_t idx = 0;
    int64_t placeX = 0;
    for (auto *rect: rectangles) {
        uint64_t prevX = rect->xCoord();
        IntegerAttr placeXAttr;
        // auto placeXAttr = IntegerAttr::get(&getContext(), placeX);
        if (idx != 0) {
            rect->xCoordAttr(placeXAttr);
        }
        placeX += prevX;
    }
}

std::unique_ptr<mlir::Pass> circt::chalk::createInitPlacementPass() {
  auto pass = std::make_unique<InitPlacementPass>();
  return pass;
}
