add_mlir_dialect_library(MLIROASIS
        OASISDialect.cpp
        OASISOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/OASIS

        DEPENDS
        MLIROASISOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
