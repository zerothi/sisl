
# In this directory we have a set of libraries
# We will need to link to the Numpy includes
set_property(DIRECTORY
  APPEND
  PROPERTY INCLUDE_DIRECTORIES
  ${NUMPY_INCLUDE_PATH}
  )

foreach(source _indices _math_small _sparse _lattice)
  add_cython_library_module(
    LIBRARY ${source}
    CYTHON ${source}_cython
    SOURCE ${source}.pyx
    )
  install(TARGETS ${source} LIBRARY DESTINATION src/sisl)
endforeach()

# Add other sub-directories
add_subdirectory("io")
add_subdirectory("physics")
