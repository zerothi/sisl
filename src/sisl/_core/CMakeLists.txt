foreach(library _lattice _dtypes _sparse)
  add_cython_library(
    SOURCE ${library}.pyx
    LIBRARY ${library}
    OUTPUT ${library}_C
    )
  install(TARGETS ${library}
    DESTINATION "${SKBUILD_PROJECT_NAME}/_core")
endforeach()
