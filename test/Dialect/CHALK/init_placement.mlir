// RUN: circt-opt --pass-pipeline='chalk.cell(chalk-init-placement)' -split-input-file %s |FileCheck %s

chalk.cell @CombNode() {
    chalk.rectangle() {xCoord = 0, yCoord = 0, width = 0, height = 0}
}
