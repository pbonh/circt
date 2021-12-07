// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-divisor-extraction)' -split-input-file %s |FileCheck %s

// circt.test copies the annotation to the target
// circt.testNT puts the targetless annotation on the circuit


// A non-local annotation should work.

// CHECK-LABEL: firrtl.circuit "CombNL"
// CHECK: firrtl.nla @nla_0 [@CombNL, @BazNL, @CombMod] ["baz", "bar", "w"]
// CHECK: firrtl.nla @nla [@CombNL, @BazNL, @CombMod] ["baz", "bar", "w2"] 
// CHECK: firrtl.module @CombMod
// CHECK: %a = firrtl.wire {annotations = [{circt.nonlocal = @nla_0, class = "circt.test", nl = "nl"}]}
// CHECK: %b = firrtl.wire {annotations = [{circt.nonlocal = @nla_0, class = "circt.test", nl = "nl"}]}
// CHECK: %w2 = firrtl.wire {annotations = [{circt.fieldID = 5 : i32, circt.nonlocal = @nla, class = "circt.test", nl = "nl2"}]} : !firrtl.bundle<a: uint, b: vector<uint, 4>> 
// CHECK: firrtl.instance bar {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}, {circt.nonlocal = @nla_0, class = "circt.nonlocal"}]} @CombMod()
// CHECK: firrtl.instance baz {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}, {circt.nonlocal = @nla_0, class = "circt.nonlocal"}]} @BazNL()
// CHECK: firrtl.module @FooL
// CHECK: %w3 = firrtl.wire {annotations = [{class = "circt.test", nl = "nl3"}]}
firrtl.circuit "CombNL"  attributes {annotations = [
  {class = "circt.test", nl = "nl", target = "~CombNL|CombNL/baz:BazNL/bar:CombMod>w"},
  {class = "circt.test", nl = "nl2", target = "~CombNL|CombNL/baz:BazNL/bar:CombMod>w2.b[2]"},
  {class = "circt.test", nl = "nl3", target = "~CombNL|FooL>w3"}
  ]}  {
  firrtl.module @CombMod() {
    %a = firrtl.wire  : !firrtl.uint<1>
    %b = firrtl.wire  : !firrtl.uint<1>
    %g = firrtl.wire  : !firrtl.uint<1>
    %h = firrtl.wire  : !firrtl.uint<1>

    %d1 = firrtl.or %a, %b : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %d2 = firrtl.or %a, %b : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %n1 = firrtl.and %g, %d1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %n2 = firrtl.and %h, %d2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  }
}


