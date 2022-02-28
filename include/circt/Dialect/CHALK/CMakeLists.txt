add_circt_dialect(CHALK chalk CHALK)
add_circt_doc(CHALK -gen-dialect-doc CHALK Dialects/)

set(LLVM_TARGET_DEFINITIONS CHALKOps.td)
mlir_tablegen(CHALKOps.h.inc -gen-enum-decls)
add_public_tablegen_target(CIRCTCHALKOpsIncGen)

#set(LLVM_TARGET_DEFINITIONS CHALKDialect.td)
#mlir_tablegen(CHALKDialect.h.inc -gen-enum-decls)
#add_public_tablegen_target(CIRCTCHALKDialectIncGen)
#add_circt_doc(CHALKDialect CHALKDialect CHALK/ -gen-dialect-doc)
#add_circt_doc(CHALKOps CHALKOps CHALK/ -gen-op-doc)
