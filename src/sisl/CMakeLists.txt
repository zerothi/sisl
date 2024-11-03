set_property(DIRECTORY
  APPEND
  PROPERTY INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/_core
  )

foreach(source _indices _math_small)
  add_cython_library(
    SOURCE ${source}.pyx
    LIBRARY ${source}
    OUTPUT ${source}_C
    )
  install(TARGETS ${source} LIBRARY
    DESTINATION ${SKBUILD_PROJECT_NAME})
endforeach()

# Python files that can be compiled with cython (Pure Python syntax)
foreach(source _sparse_grid_ops)
  sisl_compile_source(${source} compile)
  if( compile )
    add_cython_library(
      SOURCE ${source}.py
      LIBRARY ${source}
      OUTPUT ${source}_C
      )
    install(TARGETS ${source} LIBRARY
      DESTINATION ${SKBUILD_PROJECT_NAME})
  endif()
endforeach()

# We do the configure step here
# We do not have the configure file in the current directory
# Otherwise it would be installed together with it
# Also get the compile definitions
get_directory_property( SISL_DEFINITIONS DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}
  COMPILE_DEFINITIONS )

# Join to stringify list
list(JOIN SISL_DEFINITIONS " " SISL_DEFINITIONS)

configure_file(
  ${PROJECT_SOURCE_DIR}/tools/_debug_info.py.conf
  ${CMAKE_CURRENT_SOURCE_DIR}/_debug_info.py
  @ONLY
  )

# Add other sub-directories
add_subdirectory("_core")
add_subdirectory("io")
add_subdirectory("physics")
add_subdirectory("geom")
