add_mlir_dialect_library(MLIROasis
        OasisDialect.cpp
        OasisOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Oasis

        DEPENDS
        MLIROasisOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
