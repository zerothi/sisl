foreach(source _lattice _dtypes _sparse)
  add_cython_library(
    SOURCE ${source}.pyx
    LIBRARY ${source}
    OUTPUT ${source}_C
    )
  install(TARGETS ${source} LIBRARY
    DESTINATION ${SKBUILD_PROJECT_NAME}/_core)
endforeach()
