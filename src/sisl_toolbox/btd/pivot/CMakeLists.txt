
foreach(library _pivot)
  add_cython_library(
    SOURCE ${library}.pyx
    LIBRARY ${library}
    OUTPUT ${library}_C
    )
  install(TARGETS ${library}
    DESTINATION "sisl_toolbox/btd/pivot")
endforeach()
