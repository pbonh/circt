get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIROasis
        )
add_llvm_executable(oasis-opt oasis-opt.cpp)

llvm_update_compile_flags(oasis-opt)
target_link_libraries(oasis-opt PRIVATE ${LIBS})
