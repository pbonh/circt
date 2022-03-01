// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
// CHECK-LABEL:   handshake.func @simple_loop(
// CHECK-SAME:                                %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_1:.*]] = br %[[VAL_0]] : none
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = control_merge %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]]:2 = fork [2] %[[VAL_2]] : none
// CHECK:           sink %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]] = constant %[[VAL_4]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_6:.*]]:2 = fork [2] %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]] = br %[[VAL_4]]#1 : none
// CHECK:           %[[VAL_8:.*]] = br %[[VAL_6]]#0 : index
// CHECK:           %[[VAL_9:.*]] = br %[[VAL_6]]#1 : index
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = control_merge %[[VAL_7]] : none
// CHECK:           %[[VAL_12:.*]]:2 = fork [2] %[[VAL_11]] : index
// CHECK:           %[[VAL_13:.*]] = buffer [1] seq %[[VAL_14:.*]] {initValues = [0]} : i1
// CHECK:           %[[VAL_15:.*]]:3 = fork [3] %[[VAL_13]] : i1
// CHECK:           %[[VAL_16:.*]] = mux %[[VAL_15]]#2 {{\[}}%[[VAL_10]], %[[VAL_17:.*]]] : i1, none
// CHECK:           %[[VAL_18:.*]] = mux %[[VAL_12]]#1 {{\[}}%[[VAL_9]]] : index, index
// CHECK:           %[[VAL_19:.*]] = mux %[[VAL_15]]#1 {{\[}}%[[VAL_18]], %[[VAL_20:.*]]] : i1, index
// CHECK:           %[[VAL_21:.*]]:2 = fork [2] %[[VAL_19]] : index
// CHECK:           %[[VAL_22:.*]] = mux %[[VAL_12]]#0 {{\[}}%[[VAL_8]]] : index, index
// CHECK:           %[[VAL_23:.*]] = mux %[[VAL_15]]#0 {{\[}}%[[VAL_22]], %[[VAL_24:.*]]] : i1, index
// CHECK:           %[[VAL_25:.*]]:2 = fork [2] %[[VAL_23]] : index
// CHECK:           %[[VAL_14]] = merge %[[VAL_26:.*]]#0 : i1
// CHECK:           %[[VAL_27:.*]] = arith.cmpi slt, %[[VAL_25]]#0, %[[VAL_21]]#0 : index
// CHECK:           %[[VAL_26]]:4 = fork [4] %[[VAL_27]] : i1
// CHECK:           %[[VAL_28:.*]], %[[VAL_29:.*]] = cond_br %[[VAL_26]]#3, %[[VAL_21]]#1 : index
// CHECK:           sink %[[VAL_29]] : index
// CHECK:           %[[VAL_30:.*]], %[[VAL_31:.*]] = cond_br %[[VAL_26]]#2, %[[VAL_16]] : none
// CHECK:           %[[VAL_32:.*]], %[[VAL_33:.*]] = cond_br %[[VAL_26]]#1, %[[VAL_25]]#1 : index
// CHECK:           sink %[[VAL_33]] : index
// CHECK:           %[[VAL_34:.*]] = merge %[[VAL_32]] : index
// CHECK:           %[[VAL_35:.*]] = merge %[[VAL_28]] : index
// CHECK:           %[[VAL_36:.*]], %[[VAL_37:.*]] = control_merge %[[VAL_30]] : none
// CHECK:           %[[VAL_38:.*]]:2 = fork [2] %[[VAL_36]] : none
// CHECK:           sink %[[VAL_37]] : index
// CHECK:           %[[VAL_39:.*]] = constant %[[VAL_38]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_40:.*]] = arith.addi %[[VAL_34]], %[[VAL_39]] : index
// CHECK:           %[[VAL_20]] = br %[[VAL_35]] : index
// CHECK:           %[[VAL_17]] = br %[[VAL_38]]#1 : none
// CHECK:           %[[VAL_24]] = br %[[VAL_40]] : index
// CHECK:           %[[VAL_41:.*]], %[[VAL_42:.*]] = control_merge %[[VAL_31]] : none
// CHECK:           sink %[[VAL_42]] : index
// CHECK:           return %[[VAL_41]] : none
// CHECK:         }
func @simple_loop() {
^bb0:
  cf.br ^bb1
^bb1:	// pred: ^bb0
  %c1 = arith.constant 1 : index
  cf.br ^bb2(%c1 : index)
^bb2(%0: index):	// 2 preds: ^bb1, ^bb3
  %1 = arith.cmpi slt, %0, %c1 : index
  cf.cond_br %1, ^bb3, ^bb4
^bb3:	// pred: ^bb2
  %c1_0 = arith.constant 1 : index
  %2 = arith.addi %0, %c1_0 : index
  cf.br ^bb2(%2 : index)
^bb4:	// pred: ^bb2
  return
}
