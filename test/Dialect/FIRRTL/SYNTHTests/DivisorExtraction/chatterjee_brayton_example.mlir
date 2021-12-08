// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-divisor-extraction)' -split-input-file %s |FileCheck %s

// CHECK-LABEL: firrtl.circuit "CombNL"
// CHECK: firrtl.module @CombMod
// CHECK: %a = firrtl.wire {annotations = [{circt.synthID = 0 : i64}]} : !firrtl.uint<1>
// CHECK: %b = firrtl.wire {annotations = [{circt.synthID = 0 : i64}]} : !firrtl.uint<1>
// CHECK: %g = firrtl.wire {annotations = [{circt.synthID = 0 : i64}]} : !firrtl.uint<1>
// CHECK: %h = firrtl.wire {annotations = [{circt.synthID = 0 : i64}]} : !firrtl.uint<1>
// CHECK: %d = firrtl.or %a, %b : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK: %n1 = firrtl.and %g, %d {annotations = [{circt.synthID = 0 : i64}]} : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK: %n2 = firrtl.and %h, %d {annotations = [{circt.synthID = 0 : i64}]} : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
firrtl.circuit "CombNL" {
  firrtl.module @CombMod() {
    %a = firrtl.wire {annotations = [{circt.synthID = 0 : i64}]} : !firrtl.uint<1>
    %b = firrtl.wire {annotations = [{circt.synthID = 0 : i64}]} : !firrtl.uint<1>
    %g = firrtl.wire {annotations = [{circt.synthID = 0 : i64}]} : !firrtl.uint<1>
    %h = firrtl.wire {annotations = [{circt.synthID = 0 : i64}]} : !firrtl.uint<1>

    %d1 = firrtl.or %a, %b : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %d2 = firrtl.or %a, %b : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %n1 = firrtl.and %g, %d1 {annotations = [{circt.synthID = 0 : i64}]} : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %n2 = firrtl.and %h, %d2 {annotations = [{circt.synthID = 0 : i64}]} : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  }
}


