# CHALK
add_circt_dialect(CHALK chalk CHALK)
add_circt_doc(CHALK -gen-dialect-doc CHALK Dialects/)

set(LLVM_TARGET_DEFINITIONS CHALK.td)
mlir_tablegen(CHALKEnums.h.inc -gen-enum-decls)
mlir_tablegen(CHALKEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(CHALKAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(CHALKAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(CIRCTCHALKEnumsIncGen)
add_circt_doc(CHALK -gen-attrdef-doc CHALKAttributes Dialects/)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTCHALKTransformsIncGen)
add_circt_doc(Passes -gen-pass-doc CHALKPasses ./)

set(LLVM_TARGET_DEFINITIONS CHALKCanonicalization.td)
mlir_tablegen(CHALKCanonicalization.h.inc -gen-rewriters)
add_public_tablegen_target(CIRCTCHALKCanonicalizationIncGen)

set(LLVM_TARGET_DEFINITIONS CHALKOpInterfaces.td)
mlir_tablegen(CHALKOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(CHALKOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(CIRCTCHALKOpInterfacesIncGen)
add_dependencies(circt-headers CIRCTCHALKOpInterfacesIncGen)

## CHIRRTL
#add_circt_dialect(CHIRRTL chirrtl CHIRRTL)
#add_circt_doc(CHIRRTL -gen-dialect-doc CHIRRTL Dialects/)
