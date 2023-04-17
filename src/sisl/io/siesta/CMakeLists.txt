# Define all the sources required
set(siesta_sources
    io_m
    siesta_sc_off
    hsx_read hsx_write
    dm_read dm_write
    tshs_read tshs_write
    grid_read grid_write
    gf_read gf_write
    tsde_read tsde_write
    hs_read
    wfsx_read
  )
list(TRANSFORM siesta_sources PREPEND _src/)
list(TRANSFORM siesta_sources APPEND .f90)

# The module file for the signatures
set(siesta_module "_siesta")

if(WITH_FORTRAN)
  create_f2py_signature(
    SIGNATURE _siesta_signature.pyf
    MODULE ${siesta_module}
    SOURCES ${siesta_sources}
    OUTPUT _siesta_signature
    )

  add_f2py_library(
    # We have one module (io_m) and no f-files with common blocks
    # Nor do we have any f90 files with functions that should be exposed
    # Hence we only have the f90-wrapper
    WITH_F90_WRAPPER
    SIGNATURE "${_siesta_signature}"
    )

  install(TARGETS ${siesta_module} LIBRARY
    DESTINATION ${SKBUILD_PROJECT_NAME}/io/siesta)
endif()
