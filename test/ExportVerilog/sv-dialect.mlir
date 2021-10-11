// RUN: circt-opt %s -export-verilog -verify-diagnostics --lowering-options=alwaysFF,exprInEventControl | FileCheck %s --strict-whitespace

// CHECK-LABEL: module M1
// CHECK-NEXT:    #(parameter [41:0] param1) (
hw.module @M1<param1: i42>(%clock : i1, %cond : i1, %val : i8) {
  %wire42 = sv.reg : !hw.inout<i42>
  %forceWire = sv.wire sym @wire1 : !hw.inout<i1>
 
  %c11_i42 = hw.constant 11: i42
  // CHECK: localparam [41:0] param_x = 42'd11;
  %param_x = sv.localparam : i42 { value = 11: i42 }

  // CHECK: localparam [41:0] param_y = param1;
  %param_y = sv.localparam : i42 { value = #hw.param.decl.ref<"param1">: i42 }

  // CHECK:      always @(posedge clock) begin
  sv.always posedge %clock {
    // CHECK-NEXT: force forceWire = cond;
    sv.force %forceWire, %cond : i1
  // CHECK-NEXT:   `ifndef SYNTHESIS
    sv.ifdef.procedural "SYNTHESIS" {
    } else {
  // CHECK-NEXT:     if (PRINTF_COND_ & 1'bx & 1'bz & 1'bz & cond & forceWire)
      %tmp = sv.verbatim.expr "PRINTF_COND_" : () -> i1
      %verb_tmp = sv.verbatim.expr "{{0}}" : () -> i1 {symbols = [@wire1] }
      %tmp1 = sv.constantX : i1
      %tmp2 = sv.constantZ : i1
      %tmp3 = comb.and %tmp, %tmp1, %tmp2, %tmp2, %cond, %verb_tmp : i1
      sv.if %tmp3 {
  // CHECK-NEXT:       $fwrite(32'h80000002, "Hi\n");
        sv.fwrite "Hi\n"
      }

      // CHECK-NEXT: if (!(clock | cond))
      // CHECK-NEXT:   $fwrite(32'h80000002, "Bye\n");
      %tmp4 = comb.or %clock, %cond : i1
      sv.if %tmp4 {
      } else {
        sv.fwrite "Bye\n"
      }
  // CHECK-NEXT: release forceWire;
    sv.release %forceWire : !hw.inout<i1>
  // CHECK-NEXT:   `endif
  // CHECK-NEXT: end // always @(posedge)
    }
  }

  // CHECK-NEXT: always @(negedge clock) begin
  // CHECK-NEXT: end // always @(negedge)
  sv.always negedge %clock {
  }

  // CHECK-NEXT: always @(edge clock) begin
  // CHECK-NEXT: end // always @(edge)
  sv.always edge %clock {
  }

  // CHECK-NEXT: always @* begin
  // CHECK-NEXT: end // always
  sv.always {
  }

  // CHECK-NEXT: always @(posedge clock or negedge cond) begin
  // CHECK-NEXT: end // always @(posedge, negedge)
  sv.always posedge %clock, negedge %cond {
  }

  // CHECK-NEXT: always_ff @(posedge clock)
  // CHECK-NEXT:   $fwrite(32'h80000002, "Yo\n");
  sv.alwaysff(posedge %clock) {
    sv.fwrite "Yo\n"
  }

  // CHECK-NEXT: always_ff @(posedge clock) begin
  // CHECK-NEXT:   if (cond)
  // CHECK-NEXT:     $fwrite(32'h80000002, "Sync Reset Block\n")
  // CHECK-NEXT:   else
  // CHECK-NEXT:     $fwrite(32'h80000002, "Sync Main Block\n");
  // CHECK-NEXT: end // always_ff @(posedge)
  sv.alwaysff(posedge %clock) {
    sv.fwrite "Sync Main Block\n"
  } ( syncreset : posedge %cond) {
    sv.fwrite "Sync Reset Block\n"
  }

  // CHECK-NEXT: always_ff @(posedge clock or negedge cond) begin
  // CHECK-NEXT:   if (!cond)
  // CHECK-NEXT:     $fwrite(32'h80000002, "Async Reset Block\n");
  // CHECK-NEXT:   else
  // CHECK-NEXT:     $fwrite(32'h80000002, "Async Main Block\n");
  // CHECK-NEXT: end // always_ff @(posedge or negedge)
  sv.alwaysff(posedge %clock) {
    sv.fwrite "Async Main Block\n"
  } ( asyncreset : negedge %cond) {
    sv.fwrite "Async Reset Block\n"
  }

  // CHECK-NEXT:  initial begin
  sv.initial {
    // CHECK-NEXT:   if (cond)
    sv.if %cond {
      %c42 = hw.constant 42 : i42

      // CHECK-NEXT: wire42 = 42'h2A;
      sv.bpassign %wire42, %c42 : i42
    } else {
      // CHECK: wire42 = param_y;
      sv.bpassign %wire42, %param_y : i42
    }

    // CHECK-NEXT:   if (cond)
    // CHECK-NOT: begin
    sv.if %cond {
      %c42 = hw.constant 42 : i8
      %add = comb.add %val, %c42 : i8

      // CHECK-NEXT: $fwrite(32'h80000002, "Inlined! %x\n", val + 8'h2A);
      sv.fwrite "Inlined! %x\n"(%add) : i8
    }

    // begin/end required here to avoid else-confusion.

    // CHECK-NEXT:   if (cond) begin
    sv.if %cond {
      // CHECK-NEXT: if (clock)
      sv.if %clock {
        // CHECK-NEXT: $fwrite(32'h80000002, "Inside Block\n");
        sv.fwrite "Inside Block\n"
      }
      // CHECK-NEXT: end
    } else { // CHECK-NEXT: else
      // CHECK-NOT: begin
      // CHECK-NEXT: $fwrite(32'h80000002, "Else Block\n");
      sv.fwrite "Else Block\n"
    }

    // CHECK-NEXT:   if (cond) begin
    sv.if %cond {
      // CHECK-NEXT:     $fwrite(32'h80000002, "Hi\n");
      sv.fwrite "Hi\n"

      // CHECK-NEXT:     $fwrite(32'h80000002, "Bye %x\n", val + val);
      %tmp = comb.add %val, %val : i8
      sv.fwrite "Bye %x\n"(%tmp) : i8

      // CHECK-NEXT:     assert(cond);
      sv.assert %cond : i1
      // CHECK-NEXT:     assert_0: assert(cond);
      sv.assert "assert_0" %cond : i1

      // CHECK-NEXT:     assume(cond);
      sv.assume %cond : i1
      // CHECK-NEXT:     assume_0: assume(cond);
      sv.assume "assume_0" %cond : i1

      // CHECK-NEXT:     cover(cond);
      sv.cover %cond : i1
      // CHECK-NEXT:     cover_0: cover(cond);
      sv.cover "cover_0" %cond : i1

      // CHECK-NEXT:   $fatal
      sv.fatal
      // CHECK-NEXT:   $finish
      sv.finish

      // CHECK-NEXT: Emit some stuff in verilog
      // CHECK-NEXT: Great power and responsibility!
      sv.verbatim "// Emit some stuff in verilog\n// Great power and responsibility!"

      %c42 = hw.constant 42 : i8
      %add = comb.add %val, %c42 : i8
      %c42_2 = hw.constant 42 : i8
      %xor = comb.xor %val, %c42_2 : i8
      sv.verbatim "`define MACRO(a, b) a + b"
      // CHECK-NEXT: `define MACRO
      %text = sv.verbatim.expr "`MACRO({{0}}, {{1}})" (%add, %xor): (i8,i8) -> i8

      // CHECK-NEXT: $fwrite(32'h80000002, "M: %x\n", `MACRO(val + 8'h2A, val ^ 8'h2A));
      sv.fwrite "M: %x\n"(%text) : i8

    }// CHECK-NEXT:   {{end$}}
  }
  // CHECK-NEXT:  end // initial

  // CHECK-NEXT: assert property (@(posedge clock) cond);
  sv.assert.concurrent posedge %clock %cond : i1
  // CHECK-NEXT: assert_1: assert property (@(posedge clock) cond);
  sv.assert.concurrent "assert_1" posedge %clock %cond : i1

  // CHECK-NEXT: assume property (@(posedge clock) cond);
  sv.assume.concurrent posedge %clock %cond : i1
  // CHECK-NEXT: assume_1: assume property (@(posedge clock) cond);
  sv.assume.concurrent "assume_1" posedge %clock %cond : i1

  // CHECK-NEXT: cover property (@(posedge clock) cond);
  sv.cover.concurrent posedge %clock %cond : i1
  // CHECK-NEXT: cover_1: cover property (@(posedge clock) cond);
  sv.cover.concurrent "cover_1" posedge %clock %cond : i1

  // CHECK-NEXT: initial
  // CHECK-NOT: begin
  sv.initial {
    // CHECK-NEXT: $fatal
    sv.fatal
  }

  // CHECK-NEXT: initial begin
  sv.initial {
    sv.verbatim "`define THING 1"
    // CHECK-NEXT: `define THING
    %thing = sv.verbatim.expr "`THING" : () -> i42
    // CHECK-NEXT: wire42 = `THING;
    sv.bpassign %wire42, %thing : i42

    sv.ifdef.procedural "FOO" {
      // CHECK-NEXT: `ifdef FOO
      %c1 = sv.verbatim.expr "\"THING\"" : () -> i1
      sv.fwrite "%d" (%c1) : i1
      // CHECK-NEXT: fwrite(32'h80000002, "%d", "THING");
      sv.fwrite "%d" (%c1) : i1
      // CHECK-NEXT: fwrite(32'h80000002, "%d", "THING");
      // CHECK-NEXT: `endif
    }

    // CHECK-NEXT: wire42 = `THING;
    sv.bpassign %wire42, %thing : i42

    // CHECK-NEXT: casez (val)
    sv.casez %val : i8
    // CHECK-NEXT: 8'b0000001?: begin
    case b0000001x: {
      // CHECK-NEXT: $fwrite(32'h80000002, "a");
      sv.fwrite "a"
      // CHECK-NEXT: $fwrite(32'h80000002, "b");
      sv.fwrite "b"
    } // CHECK-NEXT: end

    // CHECK-NEXT: 8'b000000?1:
    // CHECK-NOT: begin
    case b000000x1: {
      // CHECK-NEXT:  $fwrite(32'h80000002, "y");
      sv.fwrite "y"
    }  // implicit yield is ok.
    // CHECK-NEXT: default:
    // CHECK-NOT: begin
    default: {
      // CHECK-NEXT:  $fwrite(32'h80000002, "z");
      sv.fwrite "z"
    } // CHECK-NEXT: endcase

   // CHECK-NEXT: casez (cond)
   sv.casez %cond : i1
   // CHECK-NEXT: 1'b0:
     case b0: {
       // CHECK-NEXT: $fwrite(32'h80000002, "zero");
       sv.fwrite "zero"
     }
     // CHECK-NEXT: 1'b1:
     case b1: {
       // CHECK-NEXT: $fwrite(32'h80000002, "one");
       sv.fwrite "one"
     } // CHECK-NEXT: endcase
  }// CHECK-NEXT:   {{end // initial$}}

  sv.ifdef "VERILATOR"  {          // CHECK-NEXT: `ifdef VERILATOR
    sv.verbatim "`define Thing2"   // CHECK-NEXT:   `define Thing2
  } else  {                        // CHECK-NEXT: `else
    sv.verbatim "`define Thing1"   // CHECK-NEXT:   `define Thing1
  }                                // CHECK-NEXT: `endif

  %add = comb.add %val, %val : i8

  // CHECK-NEXT: `define STUFF "wire42 (val + val)"
  sv.verbatim "`define STUFF \"{{0}} ({{1}})\"" (%wire42, %add) : !hw.inout<i42>, i8

  // CHECK-NEXT: `ifdef FOO
  sv.ifdef "FOO" {
    // CHECK-NEXT: wire {{.+}} = "THING";
    %c1 = sv.verbatim.expr "\"THING\"" : () -> i1

    // CHECK-NEXT: initial begin
    sv.initial {
      // CHECK-NEXT: fwrite(32'h80000002, "%d", {{.+}});
      sv.fwrite "%d" (%c1) : i1
      // CHECK-NEXT: fwrite(32'h80000002, "%d", {{.+}});
      sv.fwrite "%d" (%c1) : i1

    // CHECK-NEXT: end // initial
    }

  // CHECK-NEXT: `endif
  }
}

// CHECK-LABEL: module Aliasing(
// CHECK-NEXT:             inout [41:0] a, b, c
hw.module @Aliasing(%a : !hw.inout<i42>, %b : !hw.inout<i42>,
                      %c : !hw.inout<i42>) {

  // CHECK: alias a = b;
  sv.alias %a, %b     : !hw.inout<i42>, !hw.inout<i42>
  // CHECK: alias a = b = c;
  sv.alias %a, %b, %c : !hw.inout<i42>, !hw.inout<i42>, !hw.inout<i42>
}

hw.module @reg_0(%in4: i4, %in8: i8) -> (a: i8, b: i8) {
  // CHECK-LABEL: module reg_0(
  // CHECK-NEXT:   input  [3:0] in4,
  // CHECK-NEXT:   input  [7:0] in8,
  // CHECK-NEXT:   output [7:0] a, b);

  // CHECK-EMPTY:
  // CHECK-NEXT: reg [7:0]       myReg;
  %myReg = sv.reg : !hw.inout<i8>

  // CHECK-NEXT: reg [41:0][7:0] myRegArray1;
  %myRegArray1 = sv.reg : !hw.inout<array<42 x i8>>

  // CHECK-EMPTY:
  sv.assign %myReg, %in8 : i8        // CHECK-NEXT: assign myReg = in8;

  %subscript1 = sv.array_index_inout %myRegArray1[%in4] : !hw.inout<array<42 x i8>>, i4
  sv.assign %subscript1, %in8 : i8   // CHECK-NEXT: assign myRegArray1[in4] = in8;

  %regout = sv.read_inout %myReg : !hw.inout<i8>

  %subscript2 = sv.array_index_inout %myRegArray1[%in4] : !hw.inout<array<42 x i8>>, i4
  %memout = sv.read_inout %subscript2 : !hw.inout<i8>

  // CHECK-NEXT: assign a = myReg;
  // CHECK-NEXT: assign b = myRegArray1[in4];
  hw.output %regout, %memout : i8, i8
}

// CHECK-LABEL: issue508
// https://github.com/llvm/circt/issues/508
hw.module @issue508(%in1: i1, %in2: i1) {
  // CHECK: wire _T = in1 | in2;
  %clock = comb.or %in1, %in2 : i1

  // CHECK-NEXT: always @(posedge _T) begin
  // CHECK-NEXT: end
  sv.always posedge %clock {
  }
}

// CHECK-LABEL: exprInlineTestIssue439
// https://github.com/llvm/circt/issues/439
hw.module @exprInlineTestIssue439(%clk: i1) {
  // CHECK: always @(posedge clk) begin
  sv.always posedge %clk {
    %c = hw.constant 0 : i32

    // CHECK: localparam      [31:0] _T = 32'h0;
    // CHECK: automatic logic [15:0] _T_0 = _T[15:0];
    %e = comb.extract %c from 0 : (i32) -> i16
    %f = comb.add %e, %e : i16
    sv.fwrite "%d"(%f) : i16
    // CHECK: $fwrite(32'h80000002, "%d", _T_0 + _T_0);
    // CHECK: end // always @(posedge)
  }
}

// https://github.com/llvm/circt/issues/595
// CHECK-LABEL: module issue595
hw.module @issue595(%arr: !hw.array<128xi1>) {
  // CHECK: wire [31:0] _T;
  // CHECK: wire _T_0 = _T == 32'h0;
  %c0_i32 = hw.constant 0 : i32
  %c0_i7 = hw.constant 0 : i7
  %c0_i6 = hw.constant 0 : i6
  %0 = comb.icmp eq %3, %c0_i32 : i32

  sv.initial {
    // CHECK: assert(_T_0);
    sv.assert %0 : i1
  }

  // CHECK: wire [63:0] _T_1 = arr[7'h0+:64];
  // CHECK: assign _T = _T_1[6'h0+:32];
  %1 = hw.array_slice %arr at %c0_i7 : (!hw.array<128xi1>) -> !hw.array<64xi1>
  %2 = hw.array_slice %1 at %c0_i6 : (!hw.array<64xi1>) -> !hw.array<32xi1>
  %3 = hw.bitcast %2 : (!hw.array<32xi1>) -> i32
  hw.output
}


// CHECK-LABEL: module issue595_variant1
hw.module @issue595_variant1(%arr: !hw.array<128xi1>) {
  // CHECK: wire [31:0] _T;
  // CHECK: wire _T_0 = |_T;
  %c0_i32 = hw.constant 0 : i32
  %c0_i7 = hw.constant 0 : i7
  %c0_i6 = hw.constant 0 : i6
  %0 = comb.icmp ne %3, %c0_i32 : i32

  sv.initial {
    // CHECK: assert(_T_0);
    sv.assert %0 : i1
  }

  // CHECK: wire [63:0] _T_1 = arr[7'h0+:64];
  // CHECK: assign _T = _T_1[6'h0+:32];
  %1 = hw.array_slice %arr at %c0_i7 : (!hw.array<128xi1>) -> !hw.array<64xi1>
  %2 = hw.array_slice %1 at %c0_i6 : (!hw.array<64xi1>) -> !hw.array<32xi1>
  %3 = hw.bitcast %2 : (!hw.array<32xi1>) -> i32
  hw.output
}

// CHECK-LABEL: module issue595_variant2_checkRedunctionAnd
hw.module @issue595_variant2_checkRedunctionAnd(%arr: !hw.array<128xi1>) {
  // CHECK: wire [31:0] _T;
  // CHECK: wire _T_0 = &_T;
  %c0_i32 = hw.constant -1 : i32
  %c0_i7 = hw.constant 0 : i7
  %c0_i6 = hw.constant 0 : i6
  %0 = comb.icmp eq %3, %c0_i32 : i32

  sv.initial {
    // CHECK: assert(_T_0);
    sv.assert %0 : i1
  }

  // CHECK: wire [63:0] _T_1 = arr[7'h0+:64];
  // CHECK: assign _T = _T_1[6'h0+:32];
  %1 = hw.array_slice %arr at %c0_i7 : (!hw.array<128xi1>) -> !hw.array<64xi1>
  %2 = hw.array_slice %1 at %c0_i6 : (!hw.array<64xi1>) -> !hw.array<32xi1>
  %3 = hw.bitcast %2 : (!hw.array<32xi1>) -> i32
  hw.output
}

// CHECK-LABEL: if_multi_line_expr1
hw.module @if_multi_line_expr1(%clock: i1, %reset: i1, %really_long_port: i11) {
  %tmp6 = sv.reg  : !hw.inout<i25>

  // CHECK:      if (reset)
  // CHECK-NEXT:   tmp6 <= 25'h0;
  // CHECK-NEXT: else begin
  // CHECK-NEXT:   automatic logic [24:0] _tmp = {{..}}14{really_long_port[10]}}, really_long_port};
  // CHECK-NEXT:   tmp6 <= _tmp & 25'h3039;
  // CHECK-NEXT: end
  sv.alwaysff(posedge %clock)  {
    %0 = comb.sext %really_long_port : (i11) -> i25
  %c12345_i25 = hw.constant 12345 : i25
    %1 = comb.and %0, %c12345_i25 : i25
    sv.passign %tmp6, %1 : i25
  }(syncreset : posedge %reset)  {
    %c0_i25 = hw.constant 0 : i25
    sv.passign %tmp6, %c0_i25 : i25
  }
  hw.output
}

// CHECK-LABEL: if_multi_line_expr2
hw.module @if_multi_line_expr2(%clock: i1, %reset: i1, %really_long_port: i11) {
  %tmp6 = sv.reg  : !hw.inout<i25>

  %c12345_i25 = hw.constant 12345 : i25
  %0 = comb.sext %really_long_port : (i11) -> i25
  %1 = comb.and %0, %c12345_i25 : i25
  // CHECK:        wire [24:0] _tmp = {{..}}14{really_long_port[10]}}, really_long_port};
  // CHECK-NEXT:   wire [24:0] _T = _tmp & 25'h3039;

  // CHECK:      if (reset)
  // CHECK-NEXT:   tmp6 <= 25'h0;
  // CHECK-NEXT: else
  // CHECK-NEXT:   tmp6 <= _T;
  sv.alwaysff(posedge %clock)  {
    sv.passign %tmp6, %1 : i25
  }(syncreset : posedge %reset)  {
    %c0_i25 = hw.constant 0 : i25
    sv.passign %tmp6, %c0_i25 : i25
  }
  hw.output
}

// https://github.com/llvm/circt/issues/720
// CHECK-LABEL: module issue720(
hw.module @issue720(%clock: i1, %arg1: i1, %arg2: i1, %arg3: i1) {

  // CHECK: always @(posedge clock) begin
  sv.always posedge %clock  {
    // CHECK:   automatic logic _T = arg1 & arg2;

    // CHECK:   if (arg1)
    // CHECK:     $fatal;
    sv.if %arg1  {
      sv.fatal
    }

    // CHECK:   if (_T)
    // CHECK:     $fatal;

    //this forces a common subexpression to be output out-of-line
    %610 = comb.and %arg1, %arg2 : i1
    %611 = comb.and %arg3, %610 : i1
    sv.if %610  {
      sv.fatal
    }

    // CHECK:   if (arg3 & _T)
    // CHECK:     $fatal;
    sv.if %611  {
      sv.fatal
    }
  } // CHECK: end // always @(posedge)

  hw.output
}

// CHECK-LABEL: module issue720ifdef(
hw.module @issue720ifdef(%clock: i1, %arg1: i1, %arg2: i1, %arg3: i1) {
  // CHECK: always @(posedge clock) begin
  sv.always posedge %clock  {
    // The variable for the ifdef block needs to be emitted at the top of the
    // always block since the ifdef is transparent to verilog.

    // CHECK:    automatic logic _T;
    // CHECK:    if (arg1)
    // CHECK:      $fatal;
    sv.if %arg1  {
      sv.fatal
    }

    // CHECK:    `ifdef FUN_AND_GAMES
     sv.ifdef.procedural "FUN_AND_GAMES" {
      // This forces a common subexpression to be output out-of-line
      // CHECK:      _T = arg1 & arg2;
      // CHECK:      if (_T)
      // CHECK:        $fatal;
      %610 = comb.and %arg1, %arg2 : i1
      sv.if %610  {
        sv.fatal
      }
      // CHECK:      if (arg3 & _T)
      // CHECK:        $fatal;
      %611 = comb.and %arg3, %610 : i1
     sv.if %611  {
        sv.fatal
      }
      // CHECK:    `endif
      // CHECK:  end // always @(posedge)
    }
  }
  hw.output
}

// https://github.com/llvm/circt/issues/728

// CHECK-LABEL: module issue728(
hw.module @issue728(%clock: i1, %asdfasdfasdfasdfafa: i1, %gasfdasafwjhijjafija: i1) {
  // CHECK:  always @(posedge clock) begin
  // CHECK:    automatic logic _tmp = asdfasdfasdfasdfafa & gasfdasafwjhijjafija & asdfasdfasdfasdfafa;
  // CHECK:    automatic logic _tmp_0 = gasfdasafwjhijjafija & asdfasdfasdfasdfafa & gasfdasafwjhijjafija;
  // CHECK:    $fwrite(32'h80000002, "force output");
  // CHECK:    if (_tmp & _tmp_0)
  // CHECK:      $fwrite(32'h80000002, "this cond is split");
  // CHECK:  end // always @(posedge)
  sv.always posedge %clock  {
     sv.fwrite "force output"
     %cond = comb.and %asdfasdfasdfasdfafa, %gasfdasafwjhijjafija, %asdfasdfasdfasdfafa, %gasfdasafwjhijjafija, %asdfasdfasdfasdfafa, %gasfdasafwjhijjafija : i1
     sv.if %cond  {
       sv.fwrite "this cond is split"
     }
  }
  hw.output
}

// CHECK-LABEL: module issue728ifdef(
hw.module @issue728ifdef(%clock: i1, %asdfasdfasdfasdfafa: i1, %gasfdasafwjhijjafija: i1) {
  // CHECK: always @(posedge clock) begin
  // CHECK:      automatic logic _tmp;
  // CHECK:      automatic logic _tmp_0;
  // CHECK:    $fwrite(32'h80000002, "force output");
  // CHECK:    `ifdef FUN_AND_GAMES
  // CHECK:      _tmp = asdfasdfasdfasdfafa & gasfdasafwjhijjafija & asdfasdfasdfasdfafa;
  // CHECK:      _tmp_0 = gasfdasafwjhijjafija & asdfasdfasdfasdfafa & gasfdasafwjhijjafija;
  // CHECK:      if (_tmp & _tmp_0)
  // CHECK:        $fwrite(32'h80000002, "this cond is split");
  // CHECK:    `endif
  // CHECK: end // always @(posedge)
  sv.always posedge %clock  {
     sv.fwrite "force output"
     sv.ifdef.procedural "FUN_AND_GAMES" {
       %cond = comb.and %asdfasdfasdfasdfafa, %gasfdasafwjhijjafija, %asdfasdfasdfasdfafa, %gasfdasafwjhijjafija, %asdfasdfasdfasdfafa, %gasfdasafwjhijjafija : i1
       sv.if %cond  {
         sv.fwrite "this cond is split"
       }
     }
  }
}

// CHECK-LABEL: module alwayscombTest(
hw.module @alwayscombTest(%a: i1) -> (x: i1) {
  // CHECK: reg combWire;
  %combWire = sv.reg : !hw.inout<i1>
  // CHECK: always_comb
  sv.alwayscomb {
    // CHECK-NEXT: combWire <= a
    sv.passign %combWire, %a : i1
  }

  // CHECK: assign x = combWire;
  %out = sv.read_inout %combWire : !hw.inout<i1>
  hw.output %out : i1
}


// https://github.com/llvm/circt/issues/838
// CHECK-LABEL: module inlineProceduralWiresWithLongNames(
hw.module @inlineProceduralWiresWithLongNames(%clock: i1, %in: i1) {
  %aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = sv.wire  : !hw.inout<i1>
  %0 = sv.read_inout %aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa : !hw.inout<i1>
  %bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb = sv.wire  : !hw.inout<i1>
  %1 = sv.read_inout %bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb : !hw.inout<i1>
  %r = sv.reg  : !hw.inout<uarray<1xi1>>
  %s = sv.reg  : !hw.inout<uarray<1xi1>>
  %2 = sv.array_index_inout %r[%0] : !hw.inout<uarray<1xi1>>, i1
  %3 = sv.array_index_inout %s[%1] : !hw.inout<uarray<1xi1>>, i1
  // CHECK: always_ff
  sv.alwaysff(posedge %clock)  {
    // CHECK-NEXT: r[aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa] <= in;
    sv.passign %2, %in : i1
    // CHECK-NEXT: s[bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb] <= in;
    sv.passign %3, %in : i1
  }
}

// https://github.com/llvm/circt/issues/859
// CHECK-LABEL: module oooReg(
hw.module @oooReg(%in: i1) -> (result: i1) {
  // CHECK: wire abc;
  %0 = sv.read_inout %abc : !hw.inout<i1>

  // CHECK: assign abc = in;
  sv.assign %abc, %in : i1
  %abc = sv.wire  : !hw.inout<i1>

  // CHECK: assign result = abc;
  hw.output %0 : i1
}

// https://github.com/llvm/circt/issues/865
// CHECK-LABEL: module ifdef_beginend(
hw.module @ifdef_beginend(%clock: i1, %cond: i1, %val: i8) {
  // CHECK: always @(posedge clock) begin
  sv.always posedge %clock  {
    // CHECK-NEXT: `ifndef SYNTHESIS
    sv.ifdef.procedural "SYNTHESIS"  {
    } // CHECK-NEXT: `endif
  } // CHECK-NEXT: end
} // CHECK-NEXT: endmodule

// https://github.com/llvm/circt/issues/884
// CHECK-LABEL: module ConstResetValueMustBeInlined(
hw.module @ConstResetValueMustBeInlined(%clock: i1, %reset: i1, %d: i42) -> (q: i42) {
  %c0_i42 = hw.constant 0 : i42
  %tmp = sv.reg : !hw.inout<i42>
  // CHECK:      localparam [41:0] _T = 42'h0;
  // CHECK-NEXT: always_ff @(posedge clock or posedge reset) begin
  // CHECK-NEXT:   if (reset)
  // CHECK-NEXT:     tmp <= _T;
  sv.alwaysff(posedge %clock) {
    sv.passign %tmp, %d : i42
  } (asyncreset : posedge %reset)  {
    sv.passign %tmp, %c0_i42 : i42
  }
  %1 = sv.read_inout %tmp : !hw.inout<i42>
  hw.output %1 : i42
}

// CHECK-LABEL: module OutOfLineConstantsInAlwaysSensitivity
hw.module @OutOfLineConstantsInAlwaysSensitivity() {
  // CHECK-NEXT: localparam _T = 1'h0;
  // CHECK-NEXT: always_ff @(posedge _T)
  %clock = hw.constant 0 : i1
  sv.alwaysff(posedge %clock) {}
}

// CHECK-LABEL: module TooLongConstExpr
hw.module @TooLongConstExpr() {
  %myreg = sv.reg : !hw.inout<i4200>
  // CHECK: always @* begin
  sv.always {
    // CHECK-NEXT: localparam [4199:0] _tmp = 4200'h
    // CHECK-NEXT: myreg <= _tmp + _tmp;
    %0 = hw.constant 15894191981981165163143546843135416146464164161464654561818646486465164684484 : i4200
    %1 = comb.add %0, %0 : i4200
    sv.passign %myreg, %1 : i4200
  }
  // CHECK-NEXT: end
}

// Constants defined before use should be emitted in-place.
// CHECK-LABEL: module ConstantDefBeforeUse
hw.module @ConstantDefBeforeUse() {
  %myreg = sv.reg : !hw.inout<i32>
  // CHECK:      localparam [31:0] _T = 32'h2A;
  // CHECK-NEXT: always @*
  // CHECK-NEXT:   myreg <= _T
  %0 = hw.constant 42 : i32
  sv.always {
    sv.passign %myreg, %0 : i32
  }
}

// Constants defined after use in non-procedural regions should be moved to the
// top of the block.
// CHECK-LABEL: module ConstantDefAfterUse
hw.module @ConstantDefAfterUse() {
  %myreg = sv.reg : !hw.inout<i32>
  // CHECK:      localparam [31:0] _T = 32'h2A;
  // CHECK-NEXT: always @*
  // CHECK-NEXT:   myreg <= _T
  sv.always {
    sv.passign %myreg, %0 : i32
  }
  %0 = hw.constant 42 : i32
}

// Constants defined in a procedural block with users in a different block
// should be emitted at the top of their defining block.
// CHECK-LABEL: module ConstantEmissionAtTopOfBlock
hw.module @ConstantEmissionAtTopOfBlock() {
  %myreg = sv.reg : !hw.inout<i32>
  // CHECK:      always @* begin
  // CHECK-NEXT:   localparam [31:0] _T = 32'h2A;
  // CHECK:          myreg <= _T;
  sv.always {
    %0 = hw.constant 42 : i32
    %1 = hw.constant 1 : i1
    sv.if %1 {
      sv.passign %myreg, %0 : i32
    }
  }
}

// See https://github.com/llvm/circt/issues/1356
// CHECK-LABEL: module RegisterOfStructOrArrayOfStruct
hw.module @RegisterOfStructOrArrayOfStruct() {
  // CHECK-NOT: reg
  // CHECK: struct packed {logic a; logic b; }           reg1
  %reg1 = sv.reg : !hw.inout<struct<a: i1, b: i1>>

  // CHECK-NOT: reg
  // CHECK: struct packed {logic a; logic b; }[7:0]      reg2
  %reg2 = sv.reg : !hw.inout<array<8xstruct<a: i1, b: i1>>>

  // CHECK-NOT: reg
  // CHECK: struct packed {logic a; logic b; }[3:0][7:0] reg3
  %reg3 = sv.reg : !hw.inout<array<4xarray<8xstruct<a: i1, b: i1>>>>
}


// CHECK-LABEL: module MultiUseReadInOut(
// Issue #1564
hw.module @MultiUseReadInOut(%auto_in_ar_bits_id : i2) -> (aa: i3, bb: i3){
  %a = sv.reg  : !hw.inout<i3>
  %b = sv.reg  : !hw.inout<i3>
  %c = sv.reg  : !hw.inout<i3>
  %d = sv.reg  : !hw.inout<i3>
  %123 = sv.read_inout %b : !hw.inout<i3>
  %124 = sv.read_inout %a : !hw.inout<i3>
  %125 = sv.read_inout %c : !hw.inout<i3>
  %126 = sv.read_inout %d : !hw.inout<i3>

  // We should directly use a/b/c/d here instead of emitting temporary wires.

  // CHECK: wire [3:0][2:0] [[WIRE:.+]] = {{.}}{a}, {b}, {c}, {d}};
  // CHECK-NEXT: assign aa = [[WIRE]][auto_in_ar_bits_id];
  %127 = hw.array_create %124, %123, %125, %126 : i3
  %128 = hw.array_get %127[%auto_in_ar_bits_id] : !hw.array<4xi3>

  // CHECK: assign bb = b + a;
  %xx = comb.add %123, %124 : i3
  hw.output %128, %xx : i3, i3
}

// CHECK-LABEL: module DontDuplicateSideEffectingVerbatim(
hw.module @DontDuplicateSideEffectingVerbatim() {
  %a = sv.reg : !hw.inout<i42>
  %b = sv.reg sym @regSym : !hw.inout<i42>

  sv.initial {
    // CHECK: automatic logic [41:0] _T = SIDEEFFECT;
    %tmp = sv.verbatim.expr.se "SIDEEFFECT" : () -> i42
    // CHECK: automatic logic [41:0] _T_0 = b;
    %verb_tmp = sv.verbatim.expr.se "{{0}}" : () -> i42 {symbols = [@regSym]}
    // CHECK: a = _T;
    sv.bpassign %a, %tmp : i42
    // CHECK: a = _T;
    sv.bpassign %a, %tmp : i42

    // CHECK: a = _T_0;
    sv.bpassign %a, %verb_tmp : i42
    // CHECK: a = _T_0;
    sv.bpassign %a, %verb_tmp : i42
    %tmp2 = sv.verbatim.expr "NO_EFFECT_" : () -> i42
    // CHECK: a = NO_EFFECT_;
    sv.bpassign %a, %tmp2 : i42
    // CHECK: a = NO_EFFECT_;
    sv.bpassign %a, %tmp2 : i42
  }
}

hw.generator.schema @verbatim_schema, "Simple", ["ports", "write_latency", "read_latency"]
hw.module.extern @verbatim_inout_2 () -> ()
// CHECK-LABEL: module verbatim_M1(
hw.module @verbatim_M1(%clock : i1, %cond : i1, %val : i8) {
  %c42 = hw.constant 42 : i8
  %reg1 = sv.reg sym @verbatim_reg1: !hw.inout<i8>
  %reg2 = sv.reg sym @verbatim_reg2: !hw.inout<i8>
  %wire25 = sv.wire sym @verbatim_wireSym1 : !hw.inout<i23>
  %add = comb.add %val, %c42 : i8
  %c42_2 = hw.constant 42 : i8
  %xor = comb.xor %val, %c42_2 : i8
  hw.instance "aa1" sym @verbatim_b1 @verbatim_inout_2() ->()
  // CHECK: MACRO(val + 8'h2A, val ^ 8'h2A reg=reg1, verbatim_M2, verbatim_inout_2, aa1,reg2 = reg2 )
  sv.verbatim  "MACRO({{0}}, {{1}} reg={{2}}, {{3}}, {{4}}, {{5}},reg2 = {{6}} )" 
          (%add, %xor)  : i8,i8
          {symbols = [@verbatim_reg1, @verbatim_M2, 
          @verbatim_inout_2, @verbatim_b1, @verbatim_reg2]}
  // CHECK: Wire : wire25
  sv.verbatim " Wire : {{0}}" {symbols = [@verbatim_wireSym1]}
}

// CHECK-LABEL: module verbatim_M2(
hw.module @verbatim_M2(%clock : i1, %cond : i1, %val : i8) {
  %c42 = hw.constant 42 : i8
  %add = comb.add %val, %c42 : i8
  %c42_2 = hw.constant 42 : i8
  %xor = comb.xor %val, %c42_2 : i8
  // CHECK: MACRO(val + 8'h2A, val ^ 8'h2A, verbatim_M1 -- verbatim_M2)
  sv.verbatim  "MACRO({{0}}, {{1}}, {{2}} -- {{3}})" 
                (%add, %xor)  : i8,i8 
                {symbols = [@verbatim_M1, @verbatim_M2, @verbatim_b1]}
}

// CHECK-LABEL: module InlineAutomaticLogicInit(
// Issue #1567: https://github.com/llvm/circt/issues/1567
hw.module @InlineAutomaticLogicInit(%a : i42, %b: i42, %really_really_long_port: i11) {
  %regValue = sv.reg : !hw.inout<i42>
  // CHECK: initial begin
  sv.initial {
    // CHECK: automatic logic [63:0] _T = `THING;
    // CHECK: automatic logic [41:0] _T_0 = a + a;
    // CHECK: automatic logic [41:0] _T_1 = _T_0 + b;
    // CHECK: automatic logic [41:0] _T_2;
    %thing = sv.verbatim.expr "`THING" : () -> i64

    // CHECK: regValue = _T[44:3];
    %v = comb.extract %thing from 3 : (i64) -> i42
    sv.bpassign %regValue, %v : i42

    // tmp is multi-use, so it needs an 'automatic logic'.  This can be emitted
    // inline because it just references ports.
    %tmp = comb.add %a, %a : i42
    sv.bpassign %regValue, %tmp : i42
    // CHECK: regValue = _T_0;

    // tmp2 is as well.  This can be emitted inline because it just references
    // a port and an already-emitted-inline variable 'a'.
    %tmp2 = comb.add %tmp, %b : i42
    sv.bpassign %regValue, %tmp2 : i42
    // CHECK: regValue = _T_1;

    %tmp3 = comb.add %tmp2, %b : i42
    sv.bpassign %regValue, %tmp3 : i42
    // CHECK: regValue = _T_1 + b;

    // CHECK: `ifdef FOO
    sv.ifdef.procedural "FOO" {
      // CHECK: _T_2 = a + a;
      // tmp is multi-use so it needs a temporary, but cannot be emitted inline
      // because it is in an ifdef.
      %tmp4 = comb.add %a, %a : i42
      sv.bpassign %regValue, %tmp4 : i42
      // CHECK: regValue = _T_2;

      %tmp5 = comb.add %tmp4, %b : i42
      sv.bpassign %regValue, %tmp5 : i42
      // CHECK: regValue = _T_2 + b;
    }
  }

  // Check that inline initializer things can have too-long-line-length
  // temporaries and that they are generated correctly.
  
  // CHECK: initial begin
  sv.initial {
    // CHECK: automatic logic [41:0] [[THING:.+]] = `THING;
    // CHECK: automatic logic [41:0] _tmp = {{..}}31{really_really_long_port[10]}}, really_really_long_port};
    // CHECK: automatic logic [41:0] [[THING3:.+]] = [[THING]] + _tmp;
    // CHECK: automatic logic [41:0] _tmp_6 = [[THING]] * [[THING]]
    // CHECK: automatic logic [41:0] _tmp_7 = [[THING]] * [[THING]]
    // CHECK: automatic logic [41:0] [[MANYTHING:.+]] = _tmp_6 * _tmp_7;

    // Check the indentation level of temporaries.  Issue #1625
    // CHECK: {{^    }}automatic logic [41:0] _tmp_8;
    // CHECK: {{^    }}automatic logic [41:0] _tmp_9;
    %thing = sv.verbatim.expr.se "`THING" : () -> i42

    %thing2 = comb.sext %really_really_long_port : (i11) -> i42
    %thing3 = comb.add %thing, %thing2 : i42  // multiuse.

    // multiuse, refers to other 'automatic logic' thing so must be emitted in
    // the proper order.
    %manyThing = comb.mul %thing, %thing, %thing, %thing, %thing, %thing,
                          %thing, %thing, %thing, %thing, %thing, %thing,
                          %thing, %thing, %thing, %thing, %thing, %thing,
                          %thing, %thing, %thing, %thing, %thing, %thing : i42

    // CHECK: regValue = [[THING]];
    sv.bpassign %regValue, %thing : i42
    // CHECK: regValue = [[THING3]];
    sv.bpassign %regValue, %thing3 : i42
    // CHECK: regValue = [[THING3]];
    sv.bpassign %regValue, %thing3 : i42
    // CHECK: regValue = [[MANYTHING]];
    sv.bpassign %regValue, %manyThing : i42
    // CHECK: regValue = [[MANYTHING]];
    sv.bpassign %regValue, %manyThing : i42

    // CHECK: `ifdef FOO
    sv.ifdef.procedural "FOO" {
      sv.ifdef.procedural "BAR" {
        // Check that the temporary is inserted at the right level, not at the
        // level of the #ifdef.
        %manyMixed = comb.xor %thing, %thing, %thing, %thing, %thing, %thing,
                              %thing, %thing, %thing, %thing, %thing, %thing,
                              %thing, %thing, %thing, %thing, %thing, %thing,
                              %thing, %thing, %thing, %thing, %thing, %thing : i42
        sv.bpassign %regValue, %manyMixed : i42
      }
    }
  }
}

//CHECK-LABEL: module XMR_src
//CHECK: assign $root.a.b.c = a;
//CHECK-NEXT: assign aa = d.e.f;
hw.module @XMR_src(%a : i23) -> (aa: i3) {
  %xmr1 = sv.xmr isRooted a,b,c : !hw.inout<i23>
  %xmr2 = sv.xmr "d",e,f : !hw.inout<i3>
  %r = sv.read_inout %xmr2 : !hw.inout<i3>
  sv.assign %xmr1, %a : i23
  hw.output %r : i3
}

// CHECK-LABEL: module extInst
hw.module.extern @extInst(%_h: i1, %_i: i1, %_j: i1, %_k: i1, %_z :i0) -> ()

// CHECK-LABEL: module remoteInstDut
hw.module @remoteInstDut(%i: i1, %j: i1, %z: i0) -> () {
  %mywire = sv.wire : !hw.inout<i1>
  %mywire_rd = sv.read_inout %mywire : !hw.inout<i1>
  %myreg = sv.reg : !hw.inout<i1>
  %myreg_rd = sv.read_inout %myreg : !hw.inout<i1>
  %0 = hw.constant 1 : i1
  hw.instance "a1" sym @bindInst @extInst(_h: %mywire_rd: i1, _i: %myreg_rd: i1, _j: %j: i1, _k: %0: i1, _z: %z: i0) -> () {doNotPrint=1}
  hw.instance "a2" sym @bindInst2 @extInst(_h: %mywire_rd: i1, _i: %myreg_rd: i1, _j: %j: i1, _k: %0: i1, _z: %z: i0) -> () {doNotPrint=1}
// CHECK: wire a2__k
// CHECK-NEXT: wire a1__k
// CHECK-NEXT: wire mywire
// CHECK-NEXT: myreg
// CHECK: assign a1__k = 1'h1
// CHECK-NEXT: // This instance is elsewhere emitted as a bind statement
// CHECK-NEXT: // extInst a1
// CHECK: assign a2__k = 1'h1
// CHECK-NEXT: // This instance is elsewhere emitted as a bind statement
// CHECK-NEXT: // extInst a2
}

hw.module @bindInMod() {
  sv.bind @bindInst in @remoteInstDut
}

// CHECK-LABEL: module bindInMod();
// CHECK-NEXT:   bind remoteInstDut extInst a1 (
// CHECK-NEXT:   ._h (mywire),
// CHECK-NEXT:   ._i (myreg),
// CHECK-NEXT:   ._j (j),
// CHECK-NEXT:   ._k (a1__k)
// CHECK-NEXT: //._z (z)
// CHECK-NEXT: );
// CHECK: endmodule

sv.bind @bindInst2 in @remoteInstDut

// CHECK-LABEL: bind remoteInstDut extInst a2 (
// CHECK-NEXT:   ._h (mywire),
// CHECK-NEXT:   ._i (myreg),
// CHECK-NEXT:   ._j (j),
// CHECK-NEXT:   ._k (a2__k)
// CHECK-NEXT: //._z (z)
// CHECK-NEXT: );

