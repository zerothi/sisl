# Define all the sources required
set(sig_sources
    # We only need the precision for the signature to determine the data-precision
    # Since there are no routines, there is nothing to skip
    precision
    # Option 1: [PROPERTIES]
    io_m
    # Option 2: [DIRECT]
    #    io_m "skip: iostat_update iostat_reset :"
    siesta_sc_off
    hsx_read hsx_write
    dm_read dm_write
    tshs_read tshs_write
    grid_read grid_write
    gf_read gf_write
    tsde_read tsde_write
    wfsx_read
  )
# The REGEX allows Option 2 to pass through correctly
list(TRANSFORM sig_sources PREPEND _src/ REGEX "[^:]$")
list(TRANSFORM sig_sources APPEND .f90 REGEX "[^:]$")

# Option 1: [PROPERTIES]
# Specify the SKIP|ONLY properties to denote which
# routines to be put into the signature handling.
get_filename_component(name _src/io_m.f90 NAME)
set_source_files_properties("${name}"
  PROPERTIES SISL_SIGNATURE_SKIP "iostat_update;iostat_reset"
  )

get_filename_component(name _src/hsx_write.f90 NAME)

# Additional sources which we do not need interfaces
# to in the Python front
set(siesta_sources
  sparse_io
  )
list(TRANSFORM siesta_sources PREPEND _src/)
list(TRANSFORM siesta_sources APPEND .f90)


# The module file for the signatures
set(siesta_module "_siesta")


if(WITH_FORTRAN)
  create_f2py_signature(
    SIGNATURE _siesta_signature.pyf
    MODULE ${siesta_module}
    SOURCES ${sig_sources}
    OUTPUT _siesta_signature
    )

  add_f2py_library(
    # We have one module (io_m) and no f-files with common blocks
    # Nor do we have any f90 files with functions that should be exposed
    # Hence we only have the f90-wrapper
    WITH_F90_WRAPPER
    SIGNATURE "${_siesta_signature}"
    # In addition to the SIGNATURE sources, we also add these
    # sources (they are not part of the signature handler)
    SOURCES ${siesta_sources}
    )

  install(TARGETS ${siesta_module} LIBRARY
    DESTINATION ${SKBUILD_PROJECT_NAME}/io/siesta)
endif()
