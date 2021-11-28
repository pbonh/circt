// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-init-place)' -split-input-file %s |FileCheck %s

// circt.test copies the annotation to the target
// circt.testNT puts the targetless annotation on the circuit


// Test init placement of LoFIRRTL

// CHECK-LABEL: firrtl.circuit "CombNL"
// CHECK: firrtl.nla @nla_0 [@CombNL, @BazNL, @CombMod] ["baz", "bar", "w"]
// CHECK: firrtl.nla @nla [@CombNL, @BazNL, @CombMod] ["baz", "bar", "w2"] 
// CHECK: firrtl.module @CombMod
// CHECK: %w = firrtl.wire {annotations = [{circt.nonlocal = @nla_0, class = "circt.test", nl = "nl"}]}
// CHECK: %w2 = firrtl.wire {annotations = [{circt.fieldID = 5 : i32, circt.nonlocal = @nla, class = "circt.test", nl = "nl2"}]} : !firrtl.bundle<a: uint, b: vector<uint, 4>> 
// CHECK: firrtl.instance bar {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}, {circt.nonlocal = @nla_0, class = "circt.nonlocal"}]} @CombMod()
// CHECK: firrtl.instance baz {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}, {circt.nonlocal = @nla_0, class = "circt.nonlocal"}]} @BazNL()
// CHECK: firrtl.module @FooL
// CHECK: %w3 = firrtl.wire {annotations = [{class = "circt.test", nl = "nl3"}]}
firrtl.circuit "CombNL" {
  firrtl.module @CombMod() {
    %wa = firrtl.wire  : !firrtl.uint
    %wb = firrtl.wire  : !firrtl.uint
    %wg = firrtl.wire  : !firrtl.uint
    %wh = firrtl.wire  : !firrtl.uint
    %wn1 = firrtl.wire  : !firrtl.uint
    %wn2 = firrtl.wire  : !firrtl.uint

    %or1 = firrtl.or %wa, %wb : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint attributes {annotations = [{class = "circt.inc_place", x = "0.0", y = "0.0"}]}
    %or2 = firrtl.or %wa, %wb : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint attributes {annotations = [{class = "circt.inc_place", x = "0.0", y = "0.0"}]}
    %and1 = firrtl.and %wg, %or1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint attributes {annotations = [{class = "circt.inc_place", x = "0.0", y = "0.0"}]}
    %and2 = firrtl.and %wh, %or2 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint attributes {annotations = [{class = "circt.inc_place", x = "0.0", y = "0.0"}]}
  }
}


