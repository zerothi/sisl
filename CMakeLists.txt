cmake_minimum_required(VERSION 3.13)

project(sisl
  DESCRIPTION "Cythonized or fortran codes"
  HOMEPAGE_URL "https://github.com/zerothi/sisl"
  LANGUAGES C CXX Fortran
  )

# Enable python packages
find_package(PythonLibs REQUIRED) # libraries
find_package(PythonInterp REQUIRED) # interpreter
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)
find_package(F2PY REQUIRED)

# see https://github.com/scikit-build/scikit-build/pull/495/
set_target_properties(_f2py_runtime_library PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Print out the discovered paths
include(CMakePrintHelpers)

list(APPEND CMAKE_MESSAGE_INDENT "  ")
message(NOTICE "Python variables:")
cmake_print_variables(Python_INCLUDE_DIRS)
cmake_print_variables(F2PY_INCLUDE_DIR)
cmake_print_variables(F2PY_INCLUDE_DIRS)
cmake_print_variables(F2PY_LIBRARIES)
cmake_print_variables(Python_NumPy_INCLUDE_DIRS)
message(NOTICE "General compilation flags")
cmake_print_variables(CMAKE_C_FLAGS)
cmake_print_variables(CMAKE_CXX_FLAGS)
cmake_print_variables(CMAKE_Fortran_FLAGS)

list(POP_BACK CMAKE_MESSAGE_INDENT)

# We will never use the deprecated API any-more
add_compile_definitions(NPY_NO_DEPRECATED_API=0)
add_compile_definitions(CYTHON_NO_PYINIT_EXPORT=1)

# All libraries in sisl will *not* be prefixed with
# lib, perhaps we should change this
set(CMAKE_SHARED_MODULE_PREFIX "")

# Grab all necessary includes for numpy etc.
execute_process(
  COMMAND
  ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NUMPY_INCLUDE_PATH
  RESULT_VARIABLE NUMPY_ERR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
if(NUMPY_ERR)
  message(SEND_ERROR "WARNING: NumPy header not found.")
endif(NUMPY_ERR)


# Create a cython wrapper for creating a source and
# library
function(add_cython_library_module)
  # parse options
  set(options PY2)
  set(oneValueArgs
    LIBRARY CYTHON
    OUTPUT_VAR SOURCE)
  set(multiValueArgs)
  cmake_parse_arguments(_c "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(DEFINED _c_PY2)
    set(_c_PY "PY2")
  elseif()
    set(_c_PY "PY3")
  endif()

  # Create the cythonized sources
  add_cython_target(${_c_CYTHON} ${_c_SOURCE} ${_c_PY} OUTPUT_VAR _c_output_var)
  if( DEFINED _c_OUTPUT_VAR)
    # propagate back
    set(${_c_OUTPUT_VAR} ${_c_output_var} PARENT_SCOPE)
  endif()

  # create library
  add_library(${_c_LIBRARY} MODULE ${_c_output_var})

  # make library a python-extension (also fixes library linking)
  python_extension_module(${_c_LIBRARY})

endfunction()


# CMake + scikit-build

# Add the sisl directory (nested CMakeLists.txt files)
add_subdirectory("src")
