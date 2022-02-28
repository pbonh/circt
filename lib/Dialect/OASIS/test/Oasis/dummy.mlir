// RUN: oasis-opt %s | oasis-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = oasis.foo %{{.*}} : i32
        %res = oasis.foo %0 : i32
        return
    }
}
