# In this directory we have a set of libraries
# We will need to link to the Numpy includes
set_property(DIRECTORY
  APPEND
  PROPERTY INCLUDE_DIRECTORIES
  ${NUMPY_INCLUDE_PATH}
  )

# Define all the sources required
set(siesta_sources)
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
endforeach()

# The module file for the signatures
set(siesta_module "_siesta")

# generate the signature file
set(siesta_signature "${CMAKE_CURRENT_BINARY_DIR}/_siesta_signature.pyf")
add_custom_command(
  OUTPUT ${siesta_signature}
  COMMAND ${F2PY_EXECUTABLE} -m ${siesta_module} ${siesta_sources} -h ${siesta_signature}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${siesta_sources}
  COMMENT "Generating signature file from all siesta/_src/ files"
  )
set_source_files_properties(${siesta_signature} PROPERTIES GENERATED TRUE)

# Convert signature file to a C-module
add_f2py_target(${siesta_module} ${siesta_signature}
  DEPENDS ${siesta_signature}
  OUTPUT_VAR siesta_c
  )

foreach(file IN LISTS ${siesta_c})
  set_source_files_properties(${file} PROPERTIES GENERATED TRUE)
endforeach()

# Create siesta library with the C-interfaces + fortran sources
add_library(${siesta_module} MODULE ${siesta_c} ${siesta_sources})
target_link_libraries(${siesta_module} ${F2PY_LIBRARIES})
target_include_directories(${siesta_module} PRIVATE ${F2PY_INCLUDE_DIRS})
python_extension_module(${siesta_module})

install(TARGETS ${siesta_module} LIBRARY DESTINATION src/sisl/io/siesta)
