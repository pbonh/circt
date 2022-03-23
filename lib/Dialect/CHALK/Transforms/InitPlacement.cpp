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
#include "llvm/Support/Parallel.h"

#include <set>

using namespace circt;
using namespace chalk;

using RectangleList = std::vector< RectangleOp* >;
using RectangleOverlap = std::pair< size_t, size_t >;
using RectangleOverlaps = std::set< RectangleOverlap >;

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct InitPlacementPass : public InitPlacementBase<InitPlacementPass> {
    void runOnOperation() override;
private:
    RectangleList getRectangles();
    bool checkOverlap(RectangleOp* rect1, RectangleOp* rect2);
    RectangleOverlaps getOverlappingRectangles(const RectangleList& rectangles);
    RectangleOverlap maxOverlap(const RectangleOverlaps& overlaps);
    void fixOverlap(RectangleList& rectangles, const RectangleOverlap& maxOverlap);
};
} // end anonymous namespace

RectangleList InitPlacementPass::getRectangles() {
    auto cell = getOperation();
    RectangleList rectangles;
    llvm::for_each(cell.getBody()->getOperations(), [&](Operation &op) {
        if (isa<RectangleOp>(op)) {
            auto rectangle = dyn_cast<RectangleOp>(op);
            rectangles.push_back(&rectangle);
        }
    });
    return rectangles;
}

bool InitPlacementPass::checkOverlap(RectangleOp* rect1, RectangleOp* rect2) {
    bool overlap{false};
    struct RectDim {
        int64_t x1, x2, y1, y2;
    };
    RectDim rect1Dim, rect2Dim;
    rect1Dim.x1 = rect1->xCoord();
    rect2Dim.x1 = rect2->xCoord();
    rect1Dim.y1 = rect1->yCoord();
    rect2Dim.y1 = rect2->yCoord();
    rect1Dim.x2 = rect1Dim.x1 + rect1->width();
    rect2Dim.x2 = rect2Dim.x1 + rect2->width();
    rect1Dim.y2 = rect1Dim.y1 + rect1->height();
    rect2Dim.y2 = rect2Dim.y1 + rect2->height();
    overlap = (rect1Dim.x1 < rect2Dim.x2) && (rect1Dim.x2 > rect2Dim.x1) && (rect1Dim.y1 > rect2Dim.y2) && (rect1Dim.y2 < rect2Dim.y1);
    return overlap;
}

RectangleOverlaps InitPlacementPass::getOverlappingRectangles(const RectangleList& rectangles) {
    RectangleOverlaps overlaps;
    for (auto *rect: rectangles) {
    }
    return overlaps;
}

RectangleOverlap InitPlacementPass::maxOverlap(const RectangleOverlaps& overlaps) {
    RectangleOverlap overlap{0, 0};
    return overlap;
}

void InitPlacementPass::fixOverlap(RectangleList& rectangles, const RectangleOverlap& maxOverlap) {
}

void InitPlacementPass::runOnOperation() {
    auto rectangles = getRectangles();
    size_t idx = 0;
    int64_t prevX = 0;
    for (auto *rect: rectangles) {
        IntegerAttr prevXAttr;
        // auto prevXAttr = IntegerAttr::get(&getContext(), prevX);
        if (idx != 0) {
            rect->xCoordAttr(prevXAttr);
        }
        prevX = rect->xCoord();
    }
}

std::unique_ptr<mlir::Pass> circt::chalk::createInitPlacementPass() {
  auto pass = std::make_unique<InitPlacementPass>();
  return pass;
}
