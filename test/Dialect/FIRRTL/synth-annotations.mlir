// RUN: circt-opt %s | FileCheck %s

firrtl.circuit "And" {

// CHECK-LABEL: firrtl.module @And
firrtl.module @And(in %in1: !firrtl.uint<4>, in %in2: !firrtl.uint<4>,
                   out %out1: !firrtl.uint<4>,
                   out %out2: !firrtl.uint<4>) {
  // And operations should get CSE'd.

  // CHECK: 0 = firrtl.and %in1, %in2 {annotations = [{h = 0 : i64, l = 0 : i64, x1 = 0 : i64, x2 = 0 : i64}]} : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  %0 = firrtl.and %in1, %in2 {annotations = [{x1 = 0, x2 = 0, h = 0, l = 0}]} : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  // CHECK-NEXT: firrtl.connect %out1, %0
  firrtl.connect %out1, %0 : !firrtl.uint<4>, !firrtl.uint<4>
}
}
