add_circt_dialect(CHALK chalk)
add_circt_doc(CHALK  -gen-dialect-doc CHALK Dialects/)

set(LLVM_TARGET_DEFINITIONS CHALK.td)
mlir_tablegen(CHALKEnums.h.inc -gen-enum-decls)
mlir_tablegen(CHALKEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRCHALKEnumsIncGen)
