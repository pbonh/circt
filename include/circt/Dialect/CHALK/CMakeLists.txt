# CHALK
add_circt_dialect(CHALK chalk CHALK)
add_circt_doc(CHALK -gen-dialect-doc CHALK Dialects/)

set(LLVM_TARGET_DEFINITIONS CHALK.td)

#set(LLVM_TARGET_DEFINITIONS CHALKOps.td)
#mlir_tablegen(CHALKOps.h.inc -gen-op-interface-decls)
#mlir_tablegen(CHALKOps.cpp.inc -gen-op-interface-defs)
#add_public_tablegen_target(CIRCTCHALKOpsIncGen)
#add_dependencies(circt-headers CIRCTCHALKOpsIncGen)

#add_public_tablegen_target(CIRCTCHALKIncGen)

#set(LLVM_TARGET_DEFINITIONS CHALKDialect.td)
#mlir_tablegen(CHALKDialect.h.inc -gen-enum-decls)
#add_public_tablegen_target(CIRCTCHALKDialectIncGen)
#add_circt_doc(CHALKDialect CHALKDialect CHALK/ -gen-dialect-doc)
#add_circt_doc(CHALKOps CHALKOps CHALK/ -gen-op-doc)
