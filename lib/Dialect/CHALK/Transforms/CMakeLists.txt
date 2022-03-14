add_circt_dialect_library(CIRCTCHALKTransforms
    InitPlacement.cpp

    DEPENDS
    CIRCTCHALKTransformsIncGen

    LINK_LIBS PUBLIC
    CIRCTFIRRTL
    CIRCTHW
    CIRCTSV
    CIRCTSupport
    MLIRIR
    MLIRPass
    MLIRTransformUtils
)
