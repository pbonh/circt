// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-init-place)' -split-input-file %s |FileCheck %s

// circt.test copies the annotation to the target
// circt.testNT puts the targetless annotation on the circuit


// Test init placement of LoFIRRTL

// CHECK-LABEL: firrtl.circuit "CombNL"
// CHECK: firrtl.module @CombMod
// CHECK: %or1 = firrtl.or %wa, %wb {annotations = [{circt.xCoord = 12 : i64, circt.yCoord = 12 : i64, circt.divCost = 22 : i64}]} : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
firrtl.circuit "CombNL" {
  firrtl.module @CombMod() {
    %wa = firrtl.wire  : !firrtl.uint
    %wb = firrtl.wire  : !firrtl.uint
    %wg = firrtl.wire  : !firrtl.uint
    %wh = firrtl.wire  : !firrtl.uint
    %wn1 = firrtl.wire  : !firrtl.uint
    %wn2 = firrtl.wire  : !firrtl.uint

    %or1 = firrtl.or %wa, %wb : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint attributes {annotations = [{circt.xCoord = "0", circt.yCoord = "0"}]}
    %or2 = firrtl.or %wa, %wb : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint attributes {annotations = [{circt.xCoord = "0", circt.yCoord = "0"}]}
    %and1 = firrtl.and %wg, %or1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint attributes {annotations = [{circt.xCoord = "0", circt.yCoord = "0"}]}
    %and2 = firrtl.and %wh, %or2 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint attributes {annotations = [{circt.xCoord = "0", circt.yCoord = "0"}]}
  }
}


