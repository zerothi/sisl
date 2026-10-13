foreach(library _permute)
  add_cython_library(
    SOURCE ${library}.pyx
    LIBRARY ${library}
    OUTPUT ${library}_C
    INCLUDE_DIRS
    "${PROJECT_SOURCE_DIR}/src/sisl/_core"
    )
  install(TARGETS ${library}
    DESTINATION "sisl_toolbox/btd/permute")
endforeach()
