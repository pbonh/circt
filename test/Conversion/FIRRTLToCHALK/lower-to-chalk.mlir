// RUN: circt-opt -lower-firrtl-to-chalk %s -verify-diagnostics | FileCheck %s
// XFAIL: *

// The firrtl.circuit should NOT be removed.
// CHECK: firrtl.circuit "Simple" {

firrtl.circuit "Simple" {
  // CHECK-LABEL: firrtl.module @And(%in: i1, %sin: i1) -> (out: i1)
  firrtl.module @And(in %in: !firrtl.uint<1>,
                     in %sin: !firrtl.sint<1>,
                     out %out: !firrtl.uint<1>) attributes {annotations = [
                       {class = "chalk.firrtl", cell_name = "in", target = "~Simple|And>in"},
                       {class = "chalk.firrtl", cell_name = "sin", target = "~Simple|And>sin"},
                       {class = "chalk.firrtl", cell_name = "out", target = "~Simple|And>out"},
                     ]} {
   %out = firrtl.and %in, %sin : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  }
  // CHECK-LABEL: chalk.cell @And(%in: i1, %sin: i1) -> (out: i1)
  chalk.cell @And() {
   chalk.boundbox %in ({
   }) {
       width = 1 : !chalk.rectangle_width,
       height = 1 : !chalk.rectangle_height,
       x = 0 : !chalk.rectangle_x,
       y = 0 : !chalk.rectangle_y
       }: () -> ()
   chalk.boundbox %sin ({
   }) {
       width = 1 : !chalk.rectangle_width,
       height = 1 : !chalk.rectangle_height,
       x = 0 : !chalk.rectangle_x,
       y = 0 : !chalk.rectangle_y
       }: () -> ()
   chalk.boundbox %out ({
   }) {
       width = 1 : !chalk.rectangle_width,
       height = 1 : !chalk.rectangle_height,
       x = 0 : !chalk.rectangle_x,
       y = 0 : !chalk.rectangle_y
       }: () -> ()
  }: () -> ()
}
