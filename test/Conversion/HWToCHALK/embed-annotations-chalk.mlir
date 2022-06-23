// RUN: circt-opt %s -debug -hw-to-chalk-embed | FileCheck %s --dump-input=fail --dump-input-context=10

// CHECK-LABEL: hw.module @Foo(%ui1: i1) {
hw.module @Foo(%ui1: i1) {
    // CHECK-LABEL: %0 = comb.and %ui1, %ui1 {chalk.annotations.rectangle = {cornerline = [0, 1], x1 = 0 : i64, x2 = 0 : i64, y1 = 0 : i64, y2 = 0 : i64}} : i1
    %0 = comb.and %ui1, %ui1 {chalk.annotations.rectangle = {x1 = 0: i64, x2 = 0: i64, y1 = 0: i64, y2 = 0: i64, cornerline = [0, 1]}} : i1
}
