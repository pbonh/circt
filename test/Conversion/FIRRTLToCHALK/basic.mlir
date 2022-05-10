// RUN: circt-opt -firrtl-to-chalk -debug %s | FileCheck %s --dump-input=fail

// CHECK-LABEL: circuit Foo :
firrtl.circuit "Foo" {
  // CHECK-LABEL: module Foo :
  firrtl.module @Foo(in %ui1: !firrtl.uint<1>, out %someOut: !firrtl.uint<1>) {
    // CHECK: someOut <- ui1
    firrtl.connect %someOut, %ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    // CHECK-LABEL: %firrtl_const = firrtl.constant 42 : !firrtl.uint
    // CHECK-LABEL: %x = firrtl.node %firrtl_const : !firrtl.uint
    // CHECK-LABEL: %y = firrtl.node %firrtl_const : !firrtl.uint
    // CHECK-LABEL: %andPrimOp = firrtl.and %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint {
    // CHECK:  chalk.cell "andPrimOp" {
    // CHECK:  chalk.rectangle "andPrimOp_Rectangle" {height = 0 : ui32, width = 0 : ui32, xCoord = 0 : i64, yCoord = 0 : i64}
    // CHECK:  }
    // CHECK:  }
    %firrtl_const = firrtl.constant 42 : !firrtl.uint
    %x = firrtl.node %firrtl_const : !firrtl.uint
    %y = firrtl.node %firrtl_const : !firrtl.uint
    %andPrimOp = firrtl.and %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
  }
}
