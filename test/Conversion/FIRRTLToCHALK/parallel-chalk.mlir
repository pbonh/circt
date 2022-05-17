// RUN: circt-opt %s -firrtl-to-chalk-parallel | FileCheck %s --dump-input=fail --dump-input-context=10

// CHECK-LABEL: firrtl.circuit "Foo" {
firrtl.circuit "Foo" {
  // CHECK-LABEL: firrtl.module @Foo(in %ui1: !firrtl.uint<1>) {
  firrtl.module @Foo(in %ui1: !firrtl.uint<1>) {
    // CHECK-LABEL: %c42_ui = firrtl.constant 42 : !firrtl.uint
    // CHECK-LABEL: %x = firrtl.node %c42_ui : !firrtl.uint
    // CHECK-LABEL: %y = firrtl.node %c42_ui : !firrtl.uint
    // CHECK-LABEL: %0 = firrtl.and %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    // CHECK:  }
    // CHECK:  }
    %firrtl_const = firrtl.constant 42 : !firrtl.uint
    %x = firrtl.node %firrtl_const : !firrtl.uint
    %y = firrtl.node %firrtl_const : !firrtl.uint
    %andPrimOp = firrtl.and %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
  }
}

// CHECK-LABEL: hw.module @Foo(%ui1: i1) {
// COM:  chalk.cell "1" {
// COM:  chalk.rectangle "1_Rectangle" {height = 0 : ui32, width = 0 : ui32, xCoord = 0 : i64, yCoord = 0 : i64}
// COM:  }
// COM:  }
