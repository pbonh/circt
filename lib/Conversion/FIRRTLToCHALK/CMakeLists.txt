add_circt_conversion_library(CIRCTFIRRTLToCHALK
  FIRRTLToCHALKParallel.cpp
  FIRRTLToCHALKEmbed.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTCHALK
  CIRCTHW
  CIRCTFIRRTL
  MLIRTransforms
)
