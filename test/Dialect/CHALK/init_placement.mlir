// RUN: circt-opt --pass-pipeline='chalk.cell(chalk-init-placement)' %s |FileCheck %s

// CHECK: chalk.cell @CombNode() {
chalk.cell @CombNode() {
    chalk.rectangle {xCoord = 0 : i64, yCoord = 0 : i64, width = 0 : ui64, height = 0 : ui64}
}
