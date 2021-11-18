// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-inc-place)' -split-input-file %s |FileCheck %s

// circt.test copies the annotation to the target
// circt.testNT puts the targetless annotation on the circuit


// Test incremental placement of LoFIRRTL

// CHECK-LABEL: firrtl.circuit "CombNL"
// CHECK: firrtl.nla @nla_0 [@CombNL, @BazNL, @BarNL] ["baz", "bar", "w"]
// CHECK: firrtl.nla @nla [@CombNL, @BazNL, @BarNL] ["baz", "bar", "w2"] 
// CHECK: firrtl.module @BarNL
// CHECK: %w = firrtl.wire {annotations = [{circt.nonlocal = @nla_0, class = "circt.test", nl = "nl"}]}
// CHECK: %w2 = firrtl.wire {annotations = [{circt.fieldID = 5 : i32, circt.nonlocal = @nla, class = "circt.test", nl = "nl2"}]} : !firrtl.bundle<a: uint, b: vector<uint, 4>> 
// CHECK: firrtl.instance bar {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}, {circt.nonlocal = @nla_0, class = "circt.nonlocal"}]} @BarNL()
// CHECK: firrtl.instance baz {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}, {circt.nonlocal = @nla_0, class = "circt.nonlocal"}]} @BazNL()
// CHECK: firrtl.module @FooL
// CHECK: %w3 = firrtl.wire {annotations = [{class = "circt.test", nl = "nl3"}]}
firrtl.circuit "CombNL"  attributes {annotations = [
  {class = "circt.test", nl = "nl", target = "~CombNL|CombNL/baz:BazNL/bar:BarNL>w"},
  {class = "circt.test", nl = "nl2", target = "~CombNL|CombNL/baz:BazNL/bar:BarNL>w2.b[2]"},
  {class = "circt.test", nl = "nl3", target = "~CombNL|FooL>w3"}
  ]}  {
  firrtl.module @BarNL() {
    %w = firrtl.wire  : !firrtl.uint
    %w2 = firrtl.wire  : !firrtl.bundle<a: uint, b: vector<uint, 4>>
    firrtl.skip
  }
  firrtl.module @BazNL() {
    firrtl.instance bar @BarNL()
  }
  firrtl.module @CombNL() {
    firrtl.instance baz @BazNL()
  }
  firrtl.module @FooL() {
    %w3 = firrtl.wire: !firrtl.uint
  }
}


