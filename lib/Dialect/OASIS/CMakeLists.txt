include_directories(.)
add_circt_dialect_library(MLIROASIS
  OASISDialect.cpp
  OASISOps.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/OASIS
  
  DEPENDS
  MLIROASISIncGen
  CIRCTOASISOpsIncGen
  
  LINK_COMPONENTS
  
  Support
  
  LINK_LIBS PUBLIC
  CIRCTSupport
  CIRCTHW
  MLIRIR
  MLIRPass
  )

add_dependencies(circt-headers
  MLIROASISIncGen
  CIRCTOASISOpsIncGen
  )
