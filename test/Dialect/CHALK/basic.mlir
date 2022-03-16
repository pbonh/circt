// RUN: circt-opt --pass-pipeline='chalk.cell(chalk-init-placement)' %s | FileCheck %s --dump-input=fail
// XFAIL: true

// CHECK: chalk.cell "EmptyCell" {} {}
chalk.cell "EmptyCell" {} {}

// CHECK: chalk.cell "CombCell" {} {
// CHECK:     chalk.rectangle "CombRect1" {xCoord = 0 : i64, yCoord = 0 : i64, width = 0 : ui64, height = 0 : ui64}
// CHECK:     chalk.rectangle "CombRect2" {xCoord = 0 : i64, yCoord = 0 : i64, width = 0 : ui64, height = 0 : ui64}
// CHECK: }
chalk.cell "CombCell" {} {
    chalk.rectangle "CombRect1" {xCoord = 0 : i64, yCoord = 0 : i64, width = 0 : ui64, height = 0 : ui64}
    chalk.rectangle "CombRect2" {xCoord = 0 : i64, yCoord = 0 : i64, width = 0 : ui64, height = 0 : ui64}
}
