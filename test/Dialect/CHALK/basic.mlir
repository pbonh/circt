// RUN: circt-opt --pass-pipeline='chalk.cell(chalk-init-placement)' %s | FileCheck %s --dump-input=fail

// CHECK-LABEL: module {
// CHECK:  chalk.cell "EmptyCell"
chalk.cell "EmptyCell" {}

// CHECK:  chalk.cell "CombCell" {
// CHECK:  chalk.rectangle "CombRect1" {height = 0 : ui64, width = 0 : ui64, xCoord = 0 : i64, yCoord = 0 : i64}
// CHECK:  chalk.rectangle "CombRect2" {height = 0 : ui64, width = 0 : ui64, xCoord = 0 : i64, yCoord = 0 : i64}
// CHECK:  }
// CHECK: }
chalk.cell "CombCell" {} {
    chalk.rectangle "CombRect1" {height = 0 : ui64, width = 0 : ui64, xCoord = 0 : i64, yCoord = 0 : i64}
    chalk.rectangle "CombRect2" {height = 0 : ui64, width = 0 : ui64, xCoord = 0 : i64, yCoord = 0 : i64}
}
