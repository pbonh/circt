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

using namespace circt;
using namespace chalk;

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
    auto *ctx = cell.getContext();
    if (ctx) {
        signalPassFailure();
    }
    else {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> circt::chalk::createInitPlacementPass() {
  auto pass = std::make_unique<InitPlacementPass>();
  return pass;
}
