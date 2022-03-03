include_directories(.)
add_circt_dialect_library(MLIRCHALK
  CHALKDialect.cpp
  CHALKOps.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/CHALK
  
  DEPENDS
  MLIRCHALKIncGen
  CIRCTCHALKOpsIncGen
  
  LINK_COMPONENTS
  
  Support
  
  LINK_LIBS PUBLIC
  CIRCTSupport
  CIRCTHW
  MLIRIR
  MLIRPass
  )

add_dependencies(circt-headers
  MLIRCHALKIncGen
  CIRCTCHALKOpsIncGen
  )
