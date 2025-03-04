cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
# We will use cmake_path for file-name manipulation

list(APPEND CMAKE_MESSAGE_CONTEXT "sisl")

project(sisl
  DESCRIPTION "Cythonized or fortran codes"
  HOMEPAGE_URL "https://github.com/zerothi/sisl"
  LANGUAGES C
  # cmake requires a version of X.Y.Z or some other form, it does not accept setuptools_scm dirty tags
  #  VERSION "${SKBUILD_PROJECT_VERSION}"
  )

# Print out the discovered paths
include(CMakePrintHelpers)

# Build configuration details
message(STATUS "Build configuration information")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
cmake_print_variables(CMAKE_VERSION)
cmake_print_variables(CMAKE_BUILD_TYPE)
cmake_print_variables(CMAKE_INTERPROCEDURAL_OPTIMIZATION)
cmake_print_variables(CMAKE_POSITION_INDEPENDENT_CODE)
cmake_print_variables(CMAKE_INSTALL_PREFIX)
cmake_print_variables(CMAKE_PREFIX_PATH)
if(SKBUILD)
  message(STATUS "scikit-build-core: ${SKBUILD_CORE_VERSION}")
endif()
list(POP_BACK CMAKE_MESSAGE_INDENT)

# Enable python packages
find_package(Python COMPONENTS
  Interpreter
  Development.Module
  NumPy # needed for f2py
  REQUIRED)

# Retrive the cython executable
message(CHECK_START "Checking Cython")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

message(CHECK_START "Locating executable")
find_program(CYTHON_EXECUTABLE
  NAMES cython cython3
  NO_CACHE
  REQUIRED)
message(CHECK_PASS "${CYTHON_EXECUTABLE}")

message(CHECK_START "Checking version")
execute_process(COMMAND
  "${CYTHON_EXECUTABLE}" --version
  OUTPUT_VARIABLE CYTHON_VERSION
  RESULT_VARIABLE CYTHON_VERSION_ERR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE)

# Print it out
if(CYTHON_VERSION_ERR)
  message(CHECK_FAIL "${CYTHON_VERSION_ERR}")
else()
  message(CHECK_PASS "${CYTHON_VERSION}")
endif()
list(POP_BACK CMAKE_MESSAGE_INDENT)
message(CHECK_PASS "Found Cython")


# Define global definitions
# We will never use the deprecated API any-more
add_compile_definitions(NPY_NO_DEPRECATED_API=NPY_1_20_API_VERSION)
add_compile_definitions(CYTHON_NO_PYINIT_EXPORT=1)

# All libraries in sisl will *not* be prefixed with
#: lib, perhaps we should change this
set(CMAKE_SHARED_MODULE_PREFIX "")

# Determine whether we are in CIBUILDWHEEL
# and whether we are building for the universal target
set(_def_fortran TRUE)
#if( APPLE AND
#    "$ENV{CIBUILDWHEEL}" STREQUAL "1" AND
#    "$ENV{CIBW_ARCHS}" MATCHES "(universal2|arm64)")
#  # Disable fortran here, it shouldn't be needed
#  set(_def_fortran FALSE)
#endif()

option(WITH_FORTRAN
  "Whether to compile and ship fortran sources (default ${_def_fortran})" ${_def_fortran})

# Define all options for the user
if( WITH_FORTRAN )
  enable_language(Fortran)

  set(F2PY_REPORT_ON_ARRAY_COPY 10
    CACHE STRING
    "The minimum (element) size of arrays before warning about copies")
  option(WITH_F2PY_REPORT_COPY
    "Add instructions to track f2py-copies on routine calls use -DF2PY_REPORT_ON_ARRAY_COPY=<int> as well" FALSE)#
  option(WITH_F2PY_REPORT_EXIT
    "Report timings on routine exits" FALSE)
endif()

option(WITH_COVERAGE
  "Add instructions for coverage in Cython files" FALSE)
option(WITH_LINE_DIRECTIVES
  "Create line-directives in cythonized sources" FALSE)
option(WITH_ANNOTATE
  "Create html file outputs that can be used for figuring out performance bottlenecks in cython sources" FALSE)
option(WITH_GDB
  "Add GDB-enabled Cython sources" FALSE)

# Define which pure-python modules that should not be built
set(NO_COMPILATION ""
  CACHE STRING
  "A list of files that should not be compiled [_sparse_grid_ops|_sparse_grid_ops.py]")


message(STATUS "Compilation flags")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

set(CC $ENV{CC})
set(CFLAGS $ENV{CFLAGS})
cmake_print_variables(CC)
cmake_print_variables(CFLAGS)
cmake_print_variables(CMAKE_C_COMPILER)
cmake_print_variables(CMAKE_C_FLAGS)
cmake_print_variables(CMAKE_C_STANDARD)
set(CXX $ENV{CXX})
set(CXXFLAGS $ENV{CXXFLAGS})
cmake_print_variables(CXX)
cmake_print_variables(CXXFLAGS)
cmake_print_variables(CMAKE_CXX_COMPILER)
cmake_print_variables(CMAKE_CXX_FLAGS)
cmake_print_variables(CMAKE_CXX_STANDARD)
if(WITH_FORTRAN)
  set(FC $ENV{FC})
  set(FFLAGS $ENV{FFLAGS})
  cmake_print_variables(FC)
  cmake_print_variables(FFLAGS)
  cmake_print_variables(CMAKE_Fortran_COMPILER)
  cmake_print_variables(CMAKE_Fortran_FLAGS)
  cmake_print_variables(CMAKE_Fortran_STANDARD)
endif()
set(ARCHFLAGS $ENV{ARCHFLAGS})
cmake_print_variables(ARCHFLAGS)
if( APPLE )
  cmake_print_variables(CMAKE_OSX_ARCHITECTURES)
  cmake_print_variables(CMAKE_OSX_DEPLOYMENT_TARGET)
  cmake_print_variables(CMAKE_APPLE_ARCH_SYSROOTS)
  cmake_print_variables(CMAKE_OSX_SYSROOT_PATH)
elseif( WIN32 )
  # When building using a Python not compiled with the same compiler,
  # we then need to add this definition to pybass a Cython problem
  # See https://github.com/cython/cython/issues/3405
  add_compile_definitions(MS_WIN64)
endif()

list(POP_BACK CMAKE_MESSAGE_INDENT)


# Parse Cython global options
if(WITH_COVERAGE)
  add_compile_definitions(CYTHON_TRACE=1)
  add_compile_definitions(CYTHON_TRACE_NOGIL=1)
endif()
if(WITH_ANNOTATE)
  list(APPEND CYTHON_FLAGS --annotate)
endif()
if(WITH_LINE_DIRECTIVES)
  list(APPEND CYTHON_FLAGS --line-directives)
endif()
if(WITH_GDB)
  list(APPEND CYTHON_FLAGS --gdb)
endif()


# Decide for fortran stuff
if(WITH_FORTRAN)

  # Enable Fortran language
  enable_language(Fortran)

  # Retrieve header information
  # Define f2py executable
  set(F2PY_EXECUTABLE "${PYTHON_EXECUTABLE}" -m numpy.f2py)

  # Get numpy.f2py header
  set(Python_NumPy_F2Py_INCLUDE_DIR)
  foreach(dir IN LISTS Python_NumPy_INCLUDE_DIRS)
    if(IS_DIRECTORY "${dir}/../../f2py/src")
      set(Python_NumPy_F2Py_INCLUDE_DIR "${dir}/../../f2py/src")
      break()
    endif()
  endforeach()


  if(NOT Python_NumPy_F2Py_INCLUDE_DIR)
    message(STATUS "Could not locate f2py sources in ${NumPy_NumPy_INCLUDE_DIRS}/../../f2py/src/ trying numpy.f2py.get_include()")
    execute_process(COMMAND
      "${PYTHON_EXECUTABLE}" -c "import numpy.f2py; print(numpy.f2py.get_include())"
      OUTPUT_VARIABLE Python_NumPy_F2Py_INCLUDE_DIR
      RESULT_VARIABLE NumPy_F2Py_ERR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NumPy_F2Py_ERR)
      message(SEND_ERROR "ERROR: NumPy.F2PY header not found (${err}), looked in: ${NumPy_NumPy_INCLUDE_DIRS}/../../f2py/src.")
    endif()
  endif()

  # f2py requires a basic object
  add_library(f2py_fortranobject OBJECT "${Python_NumPy_F2Py_INCLUDE_DIR}/fortranobject.c")
  target_link_libraries(f2py_fortranobject PUBLIC Python::NumPy)
  target_include_directories(f2py_fortranobject PUBLIC "${Python_NumPy_F2Py_INCLUDE_DIR}")
  set_property(TARGET f2py_fortranobject PROPERTY POSITION_INDEPENDENT_CODE ON)
  set(F2PY_LIBRARIES f2py_fortranobject)

  if(WITH_F2PY_REPORT_EXIT)
    target_compile_definitions(f2py_fortranobject PRIVATE F2PY_REPORT_ATEXIT)
  endif()
  if(WITH_F2PY_REPORT_COPY)
    target_compile_definitions(f2py_fortranobject PRIVATE F2PY_REPORT_ON_ARRAY_COPY=${F2PY_REPORT_ON_ARRAY_COPY})
  endif()

endif(WITH_FORTRAN)


message(STATUS "Python variables:")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

cmake_print_variables(Python_INCLUDE_DIRS)
cmake_print_variables(Python_NumPy_INCLUDE_DIRS)
if(WITH_FORTRAN)
  cmake_print_variables(Python_NumPy_F2Py_INCLUDE_DIR)
endif()

list(POP_BACK CMAKE_MESSAGE_INDENT)


message(STATUS "sisl options")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

cmake_print_variables(WITH_COVERAGE)
cmake_print_variables(WITH_ANNOTATE)
cmake_print_variables(WITH_LINE_DIRECTIVES)
cmake_print_variables(WITH_GDB)
cmake_print_variables(NO_COMPILATION)

cmake_print_variables(WITH_FORTRAN)
if(WITH_FORTRAN)
  cmake_print_variables(WITH_F2PY_REPORT_COPY)
  if(WITH_F2PY_REPORT_COPY)
    cmake_print_variables(F2PY_REPORT_ON_ARRAY_COPY)
  endif()
  cmake_print_variables(WITH_F2PY_REPORT_EXIT)
endif()

list(POP_BACK CMAKE_MESSAGE_INDENT)



# Return in _result whether the _file should be built, or not
# It checks whether the file is present in the NO_COMPILATION
function(sisl_compile_source _file _result)

  # By default, we always compile everything
  set(${_result} TRUE PARENT_SCOPE)

  # Check whether the file is listed in the variable

  # First get the stem of the file path (relative to the sisl source dir)
  set(in "${CMAKE_CURRENT_SOURCE_DIR}/${_file}")
  cmake_path(RELATIVE_PATH in
    BASE_DIRECTORY "${PROJECT_SOURCE_DIR}/src"
    OUTPUT_VARIABLE base
    )
  string(REPLACE "/" ";" base_list "${base}")

  # Loop over the segments of the file
  list(LENGTH base_list len)
  math(EXPR len "${len} - 1")

  foreach(i RANGE 0 ${len})

    # Create the path from the list of base_list[i:]
    list(SUBLIST base_list ${i} -1 path)
    string(REPLACE ";" "/" path "${path}")

    # Check if the file is present in the file naming list
    list(FIND NO_COMPILATION "${path}" res1)
    list(FIND NO_COMPILATION "${path}.py" res2)

    if( res1 GREATER -1 OR res2 GREATER -1 )
      set(${_result} FALSE PARENT_SCOPE)
      break()
    endif()

  endforeach()

endfunction()



# Below we have the, a bit complicated property setup for the build-mechanism.
# It allows one to properly define how specific fortran files should be
# processed by f2py for correct signature creations.


# Document the properties that we are dealing with
# user-added properties on individual files
define_property(SOURCE
  PROPERTY SISL_SIGNATURE_SKIP
  BRIEF_DOCS "Pass which subroutines/functions to skip in the signature generation of a source file (in create_f2py_signature)"
  FULL_DOCS "See BRIEF_DOCS"
  )
define_property(SOURCE
  PROPERTY SISL_SIGNATURE_ONLY
  BRIEF_DOCS "Pass which subroutines/functions to retain (skip the rest) in the signature generation of a source file (in create_f2py_signature)"
  FULL_DOCS "See BRIEF_DOCS"
  )
define_property(SOURCE
  PROPERTY SISL_GENERATE_TARGET
  BRIEF_DOCS "The target that generates a specific file, be it Cython or f2py signature file. Can be used for dependency management"
  FULL_DOCS "See BRIEF_DOCS"
  )
define_property(SOURCE
  PROPERTY SISL_MODULE_NAME
  BRIEF_DOCS "The module name passed when the signature file was created (necessary to be equivalent in the library creation"
  FULL_DOCS "See BRIEF_DOCS"
  )
define_property(SOURCE
  PROPERTY SISL_FROM_SOURCES
  BRIEF_DOCS "The sources that generated the signature file in create_f2py_signature; automatic usage of these sources in add_f2py_library"
  FULL_DOCS "See BRIEF_DOCS"
  )



function(sisl_has_substring _bool _string _substring)
  string(FIND "${_string}" "${_substring}" index)
  if(${index} EQUAL -1)
    set(${_bool} FALSE PARENT_SCOPE)
  else()
    set(${_bool} TRUE PARENT_SCOPE)
  endif()
endfunction()

function(sisl_file_as_gentarget _name _output)
  if(IS_ABSOLUTE "${_name}")
    # shorten the generated target (easier on debugging)
    # But we retain the tree-structure of the current build directory
    cmake_path(RELATIVE_PATH _name
      BASE_DIRECTORY "${PROJECT_BINARY_DIR}"
      OUTPUT_VARIABLE _name_rel)
  else()
    set(_name_rel "${_name}")
  endif()
  string(REPLACE "/" "_" gen_signature "gen_${_name_rel}")
  string(REPLACE " " "_" gen_signature "${gen_signature}")
  set(${_output} ${gen_signature} PARENT_SCOPE)
endfunction()


# Define a function for cythonizing sources
# All source-files generated with cython are expected to depend
# on numpy, the include-folder will at least be forcefully
# added.
# We might consider adding an option to disable it, but for
# now this is fine.
function(add_cython_library)
  set(options ANNOTATE GDB LINE_DIRECTIVES CXX)
  set(oneValueArgs
    SOURCE # in
    LIBRARY # out
    # these are optional
    CYTHON_EXECUTABLE # in[opt]
    OUTPUT # out[opt]
    )
  set(multiValueArgs
    CYTHON_FLAGS # in[opt]
    INCLUDE_DIRS # in[opt]
    DEPENDS # in[opt]
    )
  # Parse arguments
  cmake_parse_arguments(_c "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Figure out the source file
  if(NOT DEFINED _c_SOURCE)
    list(GET _c_UNPARSED_ARGUMENTS 0 _c_SOURCE)
  endif()

  # Parse simple flags
  if(NOT DEFINED _c_CYTHON_EXECUTABLE)
    set(_c_CYTHON_EXECUTABLE "${CYTHON_EXECUTABLE}")
  endif()
  if(NOT DEFINED _c_CYTHON_FLAGS)
    set(_c_CYTHON_FLAGS "-3" ${CYTHON_FLAGS})
  endif()

  # Figure out the extension and convert to proper extension
  cmake_path(GET _c_SOURCE STEM LAST_ONLY _c_SOURCE_stem)

  # Define the C- source
  if( _c_CXX )
    set(_c_is_c FALSE)
    set(_output_ext "cxx")
    list(APPEND _c_CYTHON_FLAGS --cplus)
  else()
    set(_c_is_c TRUE)
    set(_output_ext "c")
  endif()
  set(_c_output_base "${_c_SOURCE_stem}.${_output_ext}")
  set(_c_output_full "${CMAKE_CURRENT_BINARY_DIR}/${_c_output_base}")

  if(DEFINED _c_OUTPUT)
    # return in output the source file
    set(${_c_OUTPUT} "${_c_output_full}" PARENT_SCOPE)
  endif()

  # storage of comment
  set(_comment "Cythonizing source ${_c_SOURCE} into ${_c_output_base}")

  if( _c_ANNOTATE )
    list(APPEND _c_CYTHON_FLAGS --annotate)
  endif()
  sisl_has_substring(_has "${_c_CYTHON_FLAGS}" "--annotate")
  if(_has)
    set(_comment "${_comment} | annotation")
  endif()

  if( _c_GDB )
    list(APPEND _c_CYTHON_FLAGS --gdb)
  endif()
  sisl_has_substring(_has "${_c_CYTHON_FLAGS}" "--gdb")
  if(_has)
    set(_comment "${_comment} | GDB")
  endif()

  if( _c_LINE_DIRECTIVES )
    list(APPEND _c_CYTHON_FLAGS --line-directives)
  endif()
  sisl_has_substring(_has "${_c_CYTHON_FLAGS}" "--line-directives")
  if(_has)
    set(_comment "${_comment} | line-directives")
  endif()

  # ensure no two flags will be present
  list(REMOVE_DUPLICATES _c_CYTHON_FLAGS)

  # Add include directories
  foreach(dir IN LISTS _c_INCLUDE_DIRS)
    if(IS_DIRECTORY "${dir}")
      list(APPEND _c_CYTHON_FLAGS "-I ${dir}")
    else()
      message(WARNING "Trying to add include dir: ${dir} but the folder does not exist?")
    endif()
  endforeach()

  # Now we have everything
  sisl_file_as_gentarget(${_c_output_full} _gen_target)
  add_custom_target(${_gen_target} DEPENDS "${_c_output_full}")
  add_custom_command(
    OUTPUT "${_c_output_full}"
    DEPENDS ${_c_SOURCE} ${_c_DEPENDS}
    COMMAND "${_c_CYTHON_EXECUTABLE}" "${_c_CYTHON_FLAGS}" --output-file "${_c_output_full}" "${_c_SOURCE}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    VERBATIM
    # this handles the spaces from the c_CYTHON_FLAGS list
    COMMAND_EXPAND_LISTS
    COMMENT "${_comment}"
    )

  # Define it has numpy
  set_source_files_properties("${_c_output_full}"
    PROPERTIES
    SISL_GENERATE_TARGET "${_gen_target}"
    INCLUDE_DIRECTORIES "${Python_NumPy_INCLUDE_DIRS}"
    )

  # now we have a source file, lets create a library if required
  if(DEFINED _c_LIBRARY)
    python_add_library(${_c_LIBRARY} WITH_SOABI
      MODULE "${_c_output_full}")
    # ensure direct dependency on the source generator (for parallel builds)
    add_dependencies(${_c_LIBRARY} ${_gen_target})
  endif()

endfunction(add_cython_library)


function(create_f2py_signature)
  set(options)
  set(oneValueArgs
    SIGNATURE # in
    MODULE # in
    F2PY_EXECUTABLE # in[opt]
    OUTPUT
    )
  set(multiValueArgs
    SOURCES # in    could be foo.f only:bar bar2:
    DEPENDS # in[opt]
    F2PY_FLAGS # in[opt]
    )
  cmake_parse_arguments(_f "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # We do not check whether the required arguments are added

  # Check arguments
  if(NOT IS_ABSOLUTE "${_f_SIGNATURE}")
    set(_f_SIGNATURE "${CMAKE_CURRENT_BINARY_DIR}/${_f_SIGNATURE}")
  endif()

  # get full relative path (from current source dir)
  cmake_path(RELATIVE_PATH _f_SIGNATURE
    OUTPUT_VARIABLE _f_signature_rel)

  # add signature file output and the module name
  list(PREPEND _f_F2PY_FLAGS "-h" "${_f_SIGNATURE}")
  list(PREPEND _f_F2PY_FLAGS "-m" "${_f_MODULE}")

  if(NOT DEFINED _f_F2PY_EXECUTABLE)
    set(_f_F2PY_EXECUTABLE "${F2PY_EXECUTABLE}")
  endif()

  # Create the full-path sources
  macro(config_prefix prefix var)
    if(${var})
      message(VERBOSE "${name}[${prefix}]: ${${var}}")
      set(${var} ${prefix}: ${${var}} :)
    else()
      # clear
      unset(${var})
    endif()
  endmacro()
  macro(get_config source)
    get_filename_component(name "${source}" NAME)
    get_source_file_property(s_skip "${name}" SISL_SIGNATURE_SKIP)
    config_prefix("skip" s_skip)
    get_source_file_property(s_only "${name}" SISL_SIGNATURE_ONLY)
    config_prefix("only" s_only)
  endmacro()
  set(_sources)
  set(_sources_all)
  foreach(s IN LISTS _f_SOURCES)
    get_config("${s}")
    if(IS_ABSOLUTE "${s}")
      list(APPEND _sources "${s}")
      list(APPEND _sources_all "${s}")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${s}")
      list(APPEND _sources "${CMAKE_CURRENT_SOURCE_DIR}/${s}")
      list(APPEND _sources_all "${CMAKE_CURRENT_SOURCE_DIR}/${s}")
    else()
      # this will add the only|skip: : segments, if present
      # Since the skip:...: is generally quoted, we will ensure
      # unpacking it.
      separate_arguments(seg UNIX_COMMAND ${s})
      list(APPEND _sources_all ${seg})
    endif()
    # Add the properties of the signature handling
    list(APPEND _sources_all ${seg} ${s_skip} ${s_only})
  endforeach()

  sisl_file_as_gentarget(${_f_SIGNATURE} _gen_target)
  add_custom_target(${_gen_target} DEPENDS "${_f_SIGNATURE}")
  add_custom_command(
    OUTPUT "${_f_SIGNATURE}"
    COMMAND "${_f_F2PY_EXECUTABLE}" "${_f_F2PY_FLAGS}" "${_sources_all}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS ${_sources} ${_f_DEPENDS}
    COMMENT "Generating signature file ${_f_signature_rel}"
    VERBATIM
    COMMAND_EXPAND_LISTS
  )

  # Set the module name property, so it is easier to deal with defaults
  set_source_files_properties("${_f_SIGNATURE}"
    PROPERTIES
    SISL_GENERATE_TARGET ${_gen_target}
    SISL_MODULE_NAME "${_f_MODULE}"
    SISL_FROM_SOURCES "${_sources}"
    GENERATED TRUE
    )

  if(DEFINED _f_OUTPUT)
    set(${_f_OUTPUT} "${_f_SIGNATURE}" PARENT_SCOPE)
  endif()

endfunction(create_f2py_signature)


# Create a library from a previously created f2py signature file.
# If this signature file has been created from `create_f2py_signature` the command is very
# simple as there are properties set on the signature file enabling full automatic discovery.
#
# Create a library from a f2py signature created file
# Note that.
# For fortran 77 files we have this:
#        Input file blah.f
#        Generates
#            blahmodule.c
#            blah-f2pywrappers.f
#    When no COMMON blocks are present only a C wrapper file is generated. Wrappers are also generated to rewrite assumed shape arrays as automatic arrays.
# For fortran 90 files we have this:
#        Input file blah.f90
#        Generates:
#            blahmodule.c
#            blah-f2pywrappers.f
#            blah-f2pywrappers2.f90
#    The f90 wrapper is used to handle code which is subdivided into modules. The f wrapper makes subroutines for functions. It rewrites assumed shape arrays as automatic arrays.
# When WITH_COMMON is present, it means that some of the f files present will have common blocks
# When WITH_FUNCTIONS is present, it means that some of the f90 files present will have functions that will be wrapped in the .f file.
# When WITHOUT_MODULES is present the f90 file will be generated
# The code will adapt which wrappers are generated
function(add_f2py_library)
  set(options WITH_F_WRAPPER WITH_F90_WRAPPER)
  set(oneValueArgs
    SIGNATURE # in
    LIBRARY # in[opt] default to module name
    MODULE # in[opt] default to get_property({SIGNATURE} SISL_MODULE_NAME)
    F2PY_EXECUTABLE # in[opt]
    F2PY_FLAGS # in[opt]
    OUTPUTS # out
    )
  set(multiValueArgs
    SOURCES # in[opt] addition to get_property({SIGNATURE} SISL_FROM_SOURCES)
    DEPENDS # in[opt]
    )
  cmake_parse_arguments(_f "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Retrive module name
  if(NOT DEFINED _f_MODULE)
    get_source_file_property(_f_MODULE "${_f_SIGNATURE}" SISL_MODULE_NAME)
  endif()
  if(NOT _f_MODULE)
    message(SEND_ERROR "add_f2py_library missing MODULE argument and/or the signature file misses the property SISL_MODULE_NAME")
  endif()
  if(NOT DEFINED _f_LIBRARY)
    set(_f_LIBRARY "${_f_MODULE}")
  endif()

  get_source_file_property(_sig_SOURCES "${_f_SIGNATURE}" SISL_FROM_SOURCES)
  # Add signature sources
  list(PREPEND _f_SOURCES ${_sig_SOURCES})
  # If the user has accidentially supplide duplicates, lets remove them
  list(REMOVE_DUPLICATES _f_SOURCES)

  # retrieve the signatures target creator
  get_source_file_property(_gen_signature_target "${_f_SIGNATURE}" SISL_GENERATE_TARGET)

  if(NOT DEFINED _f_F2PY_EXECUTABLE)
    set(_f_F2PY_EXECUTABLE "${F2PY_EXECUTABLE}")
  endif()

  # get current path relative to the project top-directory
  cmake_path(RELATIVE_PATH CMAKE_CURRENT_SOURCE_DIR
    BASE_DIRECTORY "${CMAKE_PROJECT_DIR}"
    OUTPUT_VARIABLE _current_rel_path)

  # Search for source files whether there are f or f90 files
  # and also which wrappers are generated based on input
  set(_has_f FALSE)
  set(_has_f90 FALSE)
  set(_has_unknown)
  foreach(f IN LISTS _sig_SOURCES)
    cmake_path(GET f EXTENSION LAST_ONLY ext)
    if(${ext} STREQUAL ".f")
      set(_has_f TRUE)
    elseif(${ext} STREQUAL ".f90")
      set(_has_f90 TRUE)
    else()
      list(APPEND _has_unknown ${ext})
    endif()
  endforeach()
  if(_has_unknown)
    message(WARNING "Unknown extensions (${_has_unknown}) found for library ${_f_LIBRARY} (module: ${_f_MODULE})")
  endif()


  set(_wrappers)
  if(_f_WITH_F_WRAPPER)
    list(APPEND _wrappers "${CMAKE_CURRENT_BINARY_DIR}/${_f_MODULE}-f2pywrappers.f")
  endif()
  if(_f_WITH_F90_WRAPPER)
    list(APPEND _wrappers "${CMAKE_CURRENT_BINARY_DIR}/${_f_MODULE}-f2pywrappers2.f90")
  endif()

  #set(_wrappers)
  #if(_has_f)
  #  if(_f_WITH_COMMON)
  #    list(APPEND _wrappers "${CMAKE_CURRENT_BINARY_DIR}/${_f_MODULE}-f2pywrappers.f")
  #  endif()
  #endif()
  #if(_has_f90)
  #  if(NOT _f_WITHOUT_MODULES)
  #    list(APPEND _wrappers "${CMAKE_CURRENT_BINARY_DIR}/${_f_MODULE}-f2pywrappers2.f90")
  #  endif()
  #  if(_f_WITH_FUNCTIONS)
  #    list(APPEND _wrappers "${CMAKE_CURRENT_BINARY_DIR}/${_f_MODULE}-f2pywrappers.f")
  #  endif()
  #endif()
  list(REMOVE_DUPLICATES _wrappers)


  # default to a C output, I don't think f2py can do anything differently
  set(extension "c")
  set(_module_file "${CMAKE_CURRENT_BINARY_DIR}/${_f_MODULE}module.${extension}")

  # add custom target to allow parallel builds
  sisl_file_as_gentarget("${_module_file}" _gen_target)
  add_custom_target(${_gen_target} DEPENDS "${_module_file}")
  add_custom_command(
    OUTPUT "${_module_file}"
    BYPRODUCTS "${_wrappers}"
    DEPENDS ${_f_SOURCES} ${_f_DEPENDS} "${_f_SIGNATURE}" ${_gen_signature_target}
    COMMAND "${_f_F2PY_EXECUTABLE}" "${_f_F2PY_FLAGS}" "${_f_SIGNATURE}" "--lower"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND_EXPAND_LISTS
    VERBATIM
    COMMENT "Generating module file from signature ${_f_SIGNATURE}"
  )

  python_add_library(${_f_LIBRARY} WITH_SOABI
    MODULE "${_module_file}" ${_f_SOURCES} ${_wrappers}
    )

  target_link_libraries(${_f_LIBRARY} PRIVATE ${F2PY_LIBRARIES})
  # ensure direct dependency on the source generator (for parallel builds)
  add_dependencies(${_f_LIBRARY} ${_gen_target})

  if(DEFINED _f_OUTPUTS)
    set(${_f_OUTPUTS} "${_module_file}" ${_wrappers}
      PARENT_SCOPE)
  endif()

endfunction(add_f2py_library)


# Add the sisl directory (nested CMakeLists.txt files)
add_subdirectory("src")
