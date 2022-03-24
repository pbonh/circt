// RUN: circt-opt %s --export-verilog --verify-diagnostics -o %t | FileCheck %s --strict-whitespace

// CHECK-LABEL: module AlwaysSpill(
hw.module @AlwaysSpill(%port: i1) {
  %false = hw.constant false
  %true = hw.constant true
  %awire = sv.wire : !hw.inout<i1>

  // CHECK: wire awire;
  %awire2 = sv.read_inout %awire : !hw.inout<i1>

  // Existing simple names should not cause additional spill.
  // CHECK: always @(posedge port)
  sv.always posedge %port {}
  // CHECK: always_ff @(posedge port)
  sv.alwaysff(posedge %port) {}
  // CHECK: always @(posedge awire)
  sv.always posedge %awire2 {}
  // CHECK: always_ff @(posedge awire)
  sv.alwaysff(posedge %awire2) {}

  // Constant values should cause a spill.
  // CHECK: assign [[TMP:.+]] =
  // CHECK-NEXT: always @(posedge [[TMP]])
  sv.always posedge %false {}
  // CHECK: assign [[TMP:.+]] =
  // CHECK-NEXT: always_ff @(posedge [[TMP]])
  sv.alwaysff(posedge %true) {}
}

// CHECK-LABEL: module Foo
hw.module @Foo(%clock: i1, %reset0: i1, %reset1: i1) -> () {
  %0 = comb.or %reset0, %reset1 : i1
// CHECK-NOT: _GEN_0
  sv.always posedge %0 {
    sv.if %0 {
    }
  }
}
