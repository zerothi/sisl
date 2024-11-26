# In this directory we have a set of libraries
# We will need to link to the Numpy includes
set_property(DIRECTORY
  APPEND
  PROPERTY INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../_core
  )

foreach(source
    _bloch _phase
    _matrix_utils
    _matrix_k _matrix_dk _matrix_ddk
    _matrix_phase _matrix_phase_sc _matrix_phase3
    )
  add_cython_library(
    SOURCE ${source}.pyx
    LIBRARY ${source}
    OUTPUT ${source}_C
    )
  install(TARGETS ${source} LIBRARY
    DESTINATION ${SKBUILD_PROJECT_NAME}/physics)
endforeach()
