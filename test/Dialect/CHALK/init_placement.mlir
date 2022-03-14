// RUN: circt-opt --pass-pipeline='chalk.cell(chalk-init-placement)' -split-input-file %s |FileCheck %s

// CHECK: chalk.cell @CombNode
chalk.cell @CombNode() {
    chalk.rectangle() {xCoord = 0, yCoord = 0, width = 0, height = 0}
}
