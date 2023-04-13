# In this directory we have a set of libraries
# We will need to link to the Numpy includes
set_property(DIRECTORY
  APPEND
  PROPERTY INCLUDE_DIRECTORIES
  ${NUMPY_INCLUDE_PATH}
  )

# Define all the sources required
set(siesta_sources)
set(siesta_sources_full)
foreach(f
    io_m
    siesta_sc_off
    hsx_read hsx_write
    dm_read dm_write
    tshs_read tshs_write
    grid_read grid_write
    gf_read gf_write
    tsde_read tsde_write
    hs_read
    wfsx_read)
  list(APPEND siesta_sources _src/${f}.f90)
  get_source_file_property(source_loc _src/${f}.f90 LOCATION)
  list(APPEND siesta_sources_full ${source_loc})
endforeach()

# The module file for the signatures
set(siesta_module "_siesta")

# generate the signature file
set(siesta_signature "${CMAKE_CURRENT_BINARY_DIR}/_siesta_signature.pyf")

add_custom_command(
  OUTPUT ${siesta_signature}
  COMMAND ${F2PY_EXECUTABLE} -m ${siesta_module} ${siesta_sources_full} -h ${siesta_signature}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${siesta_sources}
  COMMENT "Generating signature file from all siesta/_src/ files"
  )

# Convert signature file to a C-module
add_f2py_target(
  # NAME:
  ${siesta_module}
  # F2PYInput (actual source)
  ${siesta_signature}
  OUTPUT_VAR siesta_c
  )

# Create siesta library with the C-interfaces + fortran sources
add_library(${siesta_module} MODULE ${siesta_c} ${siesta_sources})
target_link_libraries(${siesta_module} ${F2PY_LIBRARIES})
target_include_directories(${siesta_module} PRIVATE ${F2PY_INCLUDE_DIRS})
python_extension_module(${siesta_module})

install(TARGETS ${siesta_module} LIBRARY DESTINATION src/sisl/io/siesta)
