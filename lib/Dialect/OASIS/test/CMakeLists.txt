configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(OASIS_OPT_TEST_DEPENDS
        FileCheck count not
        oasis-opt
        )

add_lit_testsuite(check-oasis-opt "Running the oasis-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${OASIS_OPT_TEST_DEPENDS}
        )
set_target_properties(check-oasis-opt PROPERTIES FOLDER "Tests")

add_lit_testsuites(OASIS_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${OASIS_OPT_TEST_DEPENDS})
