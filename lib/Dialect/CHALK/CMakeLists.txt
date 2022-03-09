add_circt_dialect_library(CIRCTCHALK
  CHALKOps.cpp
  CHALKAnalysis.cpp
  CHALKDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/CHALK

  DEPENDS
  CIRCTHW
  MLIRCHALKIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTSupport
  CIRCTHW
  MLIRIR
  MLIRPass
   )

add_dependencies(circt-headers MLIRCHALKIncGen)
