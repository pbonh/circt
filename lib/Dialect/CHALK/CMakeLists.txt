add_circt_dialect_library(CIRCTCHALK
  CHALKDialect.cpp
  CHALKOps.cpp
  CHALKTypes.cpp

  DEPENDS
  MLIRCHALKIncGen

  LINK_LIBS PUBLIC
  MLIRStandard
  MLIRIR
  )
