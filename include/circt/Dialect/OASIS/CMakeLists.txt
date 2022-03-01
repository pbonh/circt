add_circt_dialect(OASIS oasis OASIS)
add_circt_doc(OASIS -gen-dialect-doc OASIS Dialects/)

set(LLVM_TARGET_DEFINITIONS OASISOps.td)
mlir_tablegen(OASISOps.h.inc -gen-enum-decls)
add_public_tablegen_target(CIRCTOASISOpsIncGen)

#set(LLVM_TARGET_DEFINITIONS OASISDialect.td)
#mlir_tablegen(OASISDialect.h.inc -gen-enum-decls)
#add_public_tablegen_target(CIRCTOASISDialectIncGen)
#add_circt_doc(OASISDialect OASISDialect OASIS/ -gen-dialect-doc)
#add_circt_doc(OASISOps OASISOps OASIS/ -gen-op-doc)
