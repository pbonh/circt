// RUN: circt-opt %s -debug -hw-to-chalk-embed | FileCheck %s --dump-input=fail --dump-input-context=10

// CHECK-LABEL: hw.module @Foo(%ui1: i1) {
hw.module @Foo(%ui1: i1) {
    // CHECK: %0 = comb.and %ui1, %ui1 : i1 {
    // CHECK: }
    %0 = comb.and %ui1, %ui1 : i1
}
