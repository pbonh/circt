// RUN: circt-opt -export-verilog  %s | FileCheck %s

hw.module @B(%a: i1) -> () {
}

// CHECK-LABEL: hw.module @TestDupInstanceName
hw.module @TestDupInstanceName(%a: i1) {
  // CHECK: hw.instance "name"
  hw.instance "name" @B(a: %a: i1) -> ()

  // CHECK: hw.instance "name_0"
  hw.instance "name" @B(a: %a: i1) -> ()
}

// CHECK-LABEL: TestEmptyInstanceName
hw.module @TestEmptyInstanceName(%a: i1) {
  // CHECK: hw.instance "_T"
  hw.instance "" @B(a: %a: i1) -> ()

  // CHECK: hw.instance "_T_0"
  hw.instance "" @B(a: %a: i1) -> ()
}

// CHECK-LABEL: hw.module @TestInstanceNameValueConflict
hw.module @TestInstanceNameValueConflict(%a: i1) {
  // CHECK:  %name = sv.wire
  %name = sv.wire : !hw.inout<i1>
  // CHECK:  %output_0 = sv.wire
  %output = sv.wire : !hw.inout<i1>
  // CHECK:  %input_1 = sv.reg
  %input = sv.reg : !hw.inout<i1>
  // CHECK: hw.instance "name_2"
  hw.instance "name" @B(a: %a: i1) -> ()
}

// https://github.com/llvm/circt/issues/681
// Rename keywords used in variable/module names
// CHECK-LABEL: hw.module @inout_3(%inout_0: i1) -> (output_1: i1)
// CHECK-NEXT: hw.output %inout_0 : i1
hw.module @inout(%inout: i1) -> (output: i1) {
  hw.output %inout : i1
}

// https://github.com/llvm/circt/issues/681
// Rename keywords used in variable/module names
// CHECK-LABEL: hw.module @reg_4(%inout_0: i1) -> (output_1: i1)
// CHECK-NEXT: hw.output %inout_0 : i1
hw.module @reg(%inout: i1) -> (output: i1) {
  hw.output %inout : i1
}

// CHECK-LABEL: hw.module @inout_inst
// CHECK: hw.instance "foo" @inout_3
hw.module @inout_inst(%a: i1) {
  %0 = hw.instance "foo" @inout (inout: %a: i1) -> (output: i1)
}

// https://github.com/llvm/circt/issues/525
// CHECK-LABEL: hw.module @issue525(%struct_0: i2, %else_1: i2) -> (casex_2: i2)
// CHECK-NEXT: %0 = comb.add %struct_0, %else_1 : i2
hw.module @issue525(%struct: i2, %else: i2) -> (casex: i2) {
  %2 = comb.add %struct, %else : i2
  hw.output %2 : i2
}

// https://github.com/llvm/circt/issues/855
// CHECK-LABEL: hw.module @nameless_reg
// CHECK-NEXT: %_T = sv.reg : !hw.inout<i4>
hw.module @nameless_reg(%a: i1) -> () {
  %661 = sv.reg : !hw.inout<i4>
}

// CHECK-LABEL: sv.interface @output_5
sv.interface @output {
  // CHECK-NEXT: sv.interface.signal @input_0 : i1
  sv.interface.signal @input : i1
  // CHECK-NEXT: sv.interface.signal @output_1 : i1
  sv.interface.signal @output : i1
  // CHECK-NEXT: sv.interface.modport @always_2
  // CHECK-SAME: ("input" @input_0, "output" @output_1)
  sv.interface.modport @always ("input" @input, "output" @output)
}

// Instantiate a module which has had its ports renamed.
// CHECK-LABEL: hw.module @ModuleWithCollision(
// CHECK-SAME:    %reg_0: i1) -> (wire_1: i1)
hw.module @ModuleWithCollision(%reg: i1) -> (wire: i1) {
  hw.output %reg : i1
}
hw.module @InstanceWithCollisions(%a: i1) {
  hw.instance "parameter" @ModuleWithCollision(reg: %a: i1) -> (wire: i1)
}

// TODO: Renaming the above interface declarations currently does not rename
// their use in the following types.

// hw.module @InterfaceAsInstance () {
//   %0 = sv.interface.instance : !sv.interface<@output>
// }
// hw.module @InterfaceInPort (%m: !sv.modport<@output::@always>) {
// }

// This is made collide with the first renaming attempt of the `@inout` module
// above.
hw.module.extern @inout_0 () -> ()
hw.module.extern @inout_1 () -> ()
hw.module.extern @inout_2 () -> ()

hw.module.extern @module_with_bool<bparam: i1>() -> ()

// CHECK-LABEL: hw.module @parameters
// CHECK: <p1_0: i42 = 17, wire_1: i1>(%p1: i8)
hw.module @parameters<p1: i42 = 17, wire: i1>(%p1: i8) {

  // CHECK: sv.ifdef "SOMEMACRO"
  sv.ifdef "SOMEMACRO" {
    // CHECK: %local_0 = sv.localparam : i1 {value = #hw.param.decl.ref<"wire_1">}
    %local = sv.localparam : i1 { value = #hw.param.decl.ref<"wire">: i1 }

    // CHECK: = hw.param.value i1 = #hw.param.decl.ref<"wire_1">
    %0 = hw.param.value i1 = #hw.param.decl.ref<"wire">
  }

  // "wire" param getting updated should update in this instance.
  
  // CHECK: hw.instance "inst" @module_with_bool<bparam: i1 = #hw.param.decl.ref<"wire_1">>
  hw.instance "inst" @module_with_bool<bparam: i1 = #hw.param.decl.ref<"wire">>() -> ()

  // CHECK: hw.instance "inst2" @module_with_bool<bparam: i1 = #hw.param.expr.xor<#hw.param.verbatim<"wire">, true>>()
  hw.instance "inst2" @module_with_bool<bparam: i1 = #hw.param.expr.xor<#hw.param.verbatim<"wire">, true>>() -> ()
}

// CHECK-LABEL: hw.module @use_parameters
hw.module @use_parameters(%xxx: i8) {
  // CHECK: hw.instance "inst" @parameters<p1_0: i42 = 27, wire_1: i1 = false>(
  hw.instance "inst" @parameters<p1: i42 = 27, wire: i1 = 0>(p1: %xxx: i8) -> ()

  // CHECK: sv.ifdef "SOMEMACRO"
  sv.ifdef "SOMEMACRO" {
    // CHECK: %xxx_0 = sv.reg : !hw.inout<i4>
    %0 = sv.reg  { name = "xxx" } : !hw.inout<i4>
  }
}

