add_circt_conversion_library(CIRCTSynthFIRRTL
  Synth.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTFIRRTL
  MLIRTransforms
)
