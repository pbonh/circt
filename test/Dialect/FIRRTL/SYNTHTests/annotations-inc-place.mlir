// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-inc-place-div-cost)' -split-input-file %s |FileCheck %s

// circt.test copies the annotation to the target
// circt.testNT puts the targetless annotation on the circuit


// Test incremental placement of LoFIRRTL Divisor Extraction

// CHECK-LABEL: firrtl.circuit "CombNL"
// CHECK: firrtl.module @CombMod
// CHECK: %orDivisor = firrtl.or %wa, %wb {annotations = [{circt.xCoord = 12 : i64, circt.yCoord = 12 : i64, circt.xCoordRectEdge = 1 : i64, circt.yCoordRectEdge = 1 : i64, circt.xCoordCost = 22 : i64, circt.yCoordCost = 22 : i64, circt.isDivisor = true : i64}]} : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
firrtl.circuit "CombNL" {
  firrtl.module @CombMod() {
    %wa = firrtl.wire  : !firrtl.uint
    %wb = firrtl.wire  : !firrtl.uint
    %wg = firrtl.wire  : !firrtl.uint
    %wh = firrtl.wire  : !firrtl.uint
    %wn1 = firrtl.wire  : !firrtl.uint
    %wn2 = firrtl.wire  : !firrtl.uint

    %orDivisor = firrtl.or %wa, %wb : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint attributes {annotations = [{circt.xCoord = 12 : i64, circt.yCoord = 12 : i64, circt.xCoordRectEdge = 1 : i64, circt.yCoordRectEdge = 1 : i64, circt.xCoordCost = 0 : i64, circt.yCoordCost = 0 : i64, circt.isDivisor = true : i64}]}
    %and1 = firrtl.and %wg, %orDivisor : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint attributes {annotations = [{class = "circt.inc_place", x = "0.0", y = "0.0"}]}
    %and2 = firrtl.and %wh, %orDivisor : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint attributes {annotations = [{class = "circt.inc_place", x = "0.0", y = "0.0"}]}
  }
}


