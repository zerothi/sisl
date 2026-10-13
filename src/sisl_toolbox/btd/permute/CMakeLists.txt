set_property(DIRECTORY
  APPEND
  PROPERTY INCLUDE_DIRECTORIES
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../sisl/_core"
  )

foreach(library _permute)
  add_cython_library(
    SOURCE ${library}.pyx
    LIBRARY ${library}
    OUTPUT ${library}_C
    )
  install(TARGETS ${library}
    DESTINATION "sisl_toolbox/btd/permute")
endforeach()
