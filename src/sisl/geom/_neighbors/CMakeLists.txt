# In this directory we have a set of libraries
# We will need to link to the Numpy includes
foreach(source _operations)
  sisl_compile_source(${source} compile)
  if( compile )
    add_cython_library(
      SOURCE ${source}.py
      LIBRARY ${source}
      OUTPUT ${source}_C
      )
    install(TARGETS ${source} LIBRARY
      DESTINATION ${SKBUILD_PROJECT_NAME}/geom/_neighbors)
  endif()
endforeach()
