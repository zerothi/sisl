# In this directory we have a set of libraries
# We will need to link to the Numpy includes
foreach(library _operations)
  sisl_compile_source(${library} compile)
  if( compile )
    add_cython_library(
      SOURCE ${library}.py
      LIBRARY ${library}
      OUTPUT ${library}_C
      )
    install(TARGETS ${library}
      DESTINATION "${SKBUILD_PROJECT_NAME}/geom/_neighbors")
  endif()
endforeach()
