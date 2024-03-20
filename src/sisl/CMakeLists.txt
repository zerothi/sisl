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
  add_cython_library(
    SOURCE ${source}.py
    LIBRARY ${source}
    OUTPUT ${source}_C
    )
  install(TARGETS ${source} LIBRARY
    DESTINATION ${SKBUILD_PROJECT_NAME})
endforeach()

# Add other sub-directories
add_subdirectory("_core")
add_subdirectory("io")
add_subdirectory("physics")
add_subdirectory("geom")
