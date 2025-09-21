cmake_minimum_required(VERSION 3.20)

# ------------------------------------------------------------------
# Project
# ------------------------------------------------------------------
project(ad
  VERSION 0.1.0
  LANGUAGES CXX C
)

# C++ standard (use features without GNU extensions)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Position independent code (safe for shared/pybind modules)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build options
option(AD_ENABLE_OPENMP   "Enable OpenMP"                    OFF)
option(AD_ENABLE_PIQP     "Build piqp_cpp Python bindings"   ON)
option(AD_ENABLE_OSQP     "Build osqp_cpp Python bindings"   ON)
option(AD_ENABLE_QDLDL    "Build qdldl_cpp Python bindings"  ON)
option(AD_ENABLE_AMDQG    "Build amdqg Python bindings"     ON)
option(AD_ENABLE_L1CORE   "Build l1core Python bindings"     ON)

# ------------------------------------------------------------------
# Dependencies via CPM or system packages
# ------------------------------------------------------------------
# First try to include CPM if present in the repo
# ---- CPM bootstrap (before any CPMAddPackage) ----
include(cmake/CPM.cmake OPTIONAL)
# Check if CPM was included, if not, fetch and include CPM
if(NOT COMMAND CPMAddPackage)
  # Include FetchContent module
  include(FetchContent)
  # Declare CPM.cmake as a FetchContent
  FetchContent_Declare(
    CPM
    GIT_REPOSITORY https://github.com/cpm-cmake/CPM.cmake.git
    GIT_TAG v0.42.0 # You can specify a specific version of CPM.cmake
  )
  # Fetch CPM
  FetchContent_MakeAvailable(CPM)
  # Include CPM.cmake after it has been fetched
  include(${cpm_SOURCE_DIR}/cmake/CPM.cmake)
endif()


# fmt (header-only/targets)
find_package(fmt CONFIG QUIET)
if (NOT fmt_FOUND)
  message(STATUS "fmt not found via find_package; will use CPM to fetch it.")
CPMAddPackage(
  NAME fmt
  GITHUB_REPOSITORY fmtlib/fmt
  GIT_TAG 11.2.0
)
endif()

# pybind11 (preferred via package config; fall back to CPM)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  CPMAddPackage(
    NAME pybind11
    GITHUB_REPOSITORY pybind/pybind11
    GIT_TAG v2.13.6
  )
endif()

find_package(Eigen3 3.3 CONFIG QUIET)
if(NOT Eigen3_FOUND)
# Eigen setup
CPMAddPackage(
  NAME Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG master # Specify the version of Eigen you want to use
)
endif()

# Python 3 — Interpreter + Development
find_package(Python3 3.13 COMPONENTS Interpreter Development REQUIRED)


    message(STATUS "nanobind not found via find_package; will use CPM to fetch it.")
    
    # CRITICAL: Set Python variables BEFORE CPMAddPackage
    # This ensures nanobind's CMake configuration uses the correct Python
    set(Python_EXECUTABLE ${Python3_EXECUTABLE} CACHE FILEPATH "" FORCE)
    set(Python3_EXECUTABLE ${Python3_EXECUTABLE} CACHE FILEPATH "" FORCE)
    set(Python_INCLUDE_DIRS ${Python3_INCLUDE_DIRS} CACHE PATH "" FORCE)
    set(Python3_INCLUDE_DIRS ${Python3_INCLUDE_DIRS} CACHE PATH "" FORCE)
    set(Python_LIBRARIES ${Python3_LIBRARIES} CACHE FILEPATH "" FORCE)
    set(Python3_LIBRARIES ${Python3_LIBRARIES} CACHE FILEPATH "" FORCE)
    
    # Also set the specific variables nanobind looks for
    set(Python_INCLUDE_DIR ${Python3_INCLUDE_DIRS} CACHE PATH "" FORCE)
    set(PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS} CACHE PATH "" FORCE)
    set(PYTHON_LIBRARIES ${Python3_LIBRARIES} CACHE FILEPATH "" FORCE)

    # allow overriding the tag on the command line if desired
    set(NANOBIND_TAG v2.9.2 CACHE STRING "nanobind version/tag to fetch via CPM")
    
    CPMAddPackage(
        NAME nanobind
        GITHUB_REPOSITORY wjakob/nanobind
        GIT_TAG ${NANOBIND_TAG}
        OPTIONS
            "Python_EXECUTABLE ${Python3_EXECUTABLE}"
            "Python3_EXECUTABLE ${Python3_EXECUTABLE}"
            "Python_INCLUDE_DIRS ${Python3_INCLUDE_DIRS}"
            "Python3_INCLUDE_DIRS ${Python3_INCLUDE_DIRS}"
            "Python_LIBRARIES ${Python3_LIBRARIES}"
            "Python3_LIBRARIES ${Python3_LIBRARIES}"
    )
    
    # Verify that nanobind found Python correctly
    if(TARGET nanobind-static)
        get_target_property(NB_INCLUDES nanobind-static INCLUDE_DIRECTORIES)
        message(STATUS "nanobind-static include directories: ${NB_INCLUDES}")
        
        # Force Python include dirs on nanobind-static if they're missing
        if(Python3_INCLUDE_DIRS AND NOT "${NB_INCLUDES}" MATCHES "Python")
            message(STATUS "Force-adding Python includes to nanobind-static")
            target_include_directories(nanobind-static PRIVATE ${Python3_INCLUDE_DIRS})
        endif()
    endif()


set(Python_EXECUTABLE      ${Python3_EXECUTABLE})
set(Python_INCLUDE_DIR     ${Python3_INCLUDE_DIRS})
set(Python_LIBRARY         ${Python3_LIBRARIES})
set(Python_NumPy_INCLUDE_DIRS ${Python3_NumPy_INCLUDE_DIRS})

# OpenMP (optional)
if(AD_ENABLE_OPENMP)
  find_package(OpenMP QUIET)
endif()

# ------------------------------------------------------------------
# Optional: GUROBI
# ------------------------------------------------------------------
# You can pass -DGUROBI_ROOT=/path/to/gurobi/linux64 on the cmake command line
set(GUROBI_ROOT "" CACHE PATH "Path to GUROBI linux64 installation root")

function(_ad_link_gurobi tgt)
  if(NOT AD_ENABLE_GUROBI)
    return()
  endif()

  if(NOT GUROBI_ROOT)
    # Default to the path you used previously; make it easy to override
    set(GUROBI_ROOT "/home/seman/gurobi1101/linux64" CACHE PATH "" FORCE)
  endif()

  set(_gurobi_inc "${GUROBI_ROOT}/include")
  # Prefer shared .so, but allow static C++ archive
  set(_gurobi_lib "${GUROBI_ROOT}/lib")

  if(EXISTS "${_gurobi_inc}" AND EXISTS "${_gurobi_lib}")
    target_include_directories(${tgt} PRIVATE "${_gurobi_inc}")
    # Typical names: libgurobi110.so + libgurobi_c++.a (matches your original)
    target_link_libraries(${tgt} PRIVATE
      "${_gurobi_lib}/libgurobi_c++.a"
      "${_gurobi_lib}/libgurobi110.so"
    )
    # Ensure runtime can find libgurobi*.so
    if(UNIX AND NOT APPLE)
      set_property(TARGET ${tgt} APPEND PROPERTY
        BUILD_RPATH "${_gurobi_lib}")
      set_property(TARGET ${tgt} APPEND PROPERTY
        INSTALL_RPATH "${_gurobi_lib}")
    endif()
  else()
    message(WARNING "GUROBI not found at GUROBI_ROOT='${GUROBI_ROOT}'. "
                    "To disable, set -DAD_ENABLE_GUROBI=OFF.")
  endif()
endfunction()

# ------------------------------------------------------------------
# Helper: add a pybind11 module with common settings
# ------------------------------------------------------------------
function(ad_add_pybind_module name)
    # Usage: ad_add_pybind_module(<name> src1.cpp src2.cpp ...)
    set(_srcs ${ARGN})  # ARGN contains all arguments after 'name'
    
    # Only need ONE of these - pybind11_add_module creates the target
    pybind11_add_module(${name} ${_srcs})
    message(STATUS "Added pybind11 module: ${name} with sources: ${_srcs}")
    
    target_compile_features(${name} PRIVATE cxx_std_23)
    
    # Common include dirs (PRIVATE)
    target_include_directories(${name}
        PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
    )
    
    # Common links
    target_link_libraries(${name}
        PRIVATE
        fmt::fmt
        Eigen3::Eigen
    )
    
    if(AD_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    
    # Nice warnings
    if(MSVC)
        target_compile_options(${name} PRIVATE /W4)
    else()
        target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

# ------------------------------------------------------------------
# Helper: add a nanobind module with common settings
# ------------------------------------------------------------------
function(ad_add_nanobind_module name)
    # Usage: ad_add_nanobind_module(<name> src1.cpp src2.cpp ...)
    set(_srcs ${ARGN})

    # Create the nanobind extension target
    nanobind_add_module(${name} ${_srcs})
    message(STATUS "Added nanobind module: ${name} with sources: ${_srcs}")

    target_compile_features(${name} PRIVATE cxx_std_23)

    # Common include dirs
    target_include_directories(${name}
        PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
        ${Python3_INCLUDE_DIRS}
    )

    # Common links
    target_link_libraries(${name}
        PRIVATE
        fmt::fmt
        Eigen3::Eigen
        Python3::Module          # provided by find_package(Python3 ... Development.Module)
    )

    if(AD_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # Warnings / visibility similar to your pybind helper
    if(MSVC)
        target_compile_options(${name} PRIVATE /W4)
    else()
        target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic -fvisibility=hidden)
    endif()
endfunction()


# ------------------------------------------------------------------
# Core AD Python module
# ------------------------------------------------------------------
# Source list isolated; add/remove files here
set(AD_SOURCES
  src/ad_bindings.cpp
  src/ad/Variable.cpp
  src/ad/Expression.cpp
  src/ad/ADGraph.cpp
)

ad_add_nanobind_module(ad ${AD_SOURCES})

# External system libs specific to 'ad' (from original list)
# Prefer target_link_libraries with plain names; avoid raw -L/-l unless needed.
# If the system requires explicit search paths, you can add them via
# target_link_directories(ad PRIVATE /usr/lib64 /usr/lib64/x86_64-linux-gnu)
# but try to rely on the linker defaults first.
target_link_libraries(ad PRIVATE
  m
  dl
)

set(L1_SOURCES
        l1penalty/src/bindings.cpp
        l1penalty/src/Solvers.cpp
        l1penalty/src/PolynomialVector.cpp
        l1penalty/src/pointWork.cpp
        l1penalty/src/Overloads.cpp
        l1penalty/src/l1Solver.cpp
        l1penalty/src/Helpers.cpp
        l1penalty/src/l1Fun.cpp
        l1penalty/src/TRModel.cpp
)

# ------------------------------------------------------------------
# Optional extra Python modules
# ------------------------------------------------------------------
if(AD_ENABLE_L1CORE)
  ad_add_pybind_module(l1core ${L1_SOURCES})
  ad_add_pybind_module(simplex_core src/simplex_bindings.cpp)
endif()

if(AD_ENABLE_PIQP)
  ad_add_pybind_module(piqp_cpp src/piqp_bindings.cpp)
endif()

if(AD_ENABLE_AMDQG)
  ad_add_pybind_module(amdqg src/amdqg_bindings.cpp)
endif()

if(AD_ENABLE_OSQP)
  # osqp_cpp
  ad_add_pybind_module(osqp_cpp src/osqp_bindings.cpp)
endif()

if(AD_ENABLE_QDLDL)
  # qdldl_cpp
  ad_add_pybind_module(qdldl_cpp src/qdldl_bindings.cpp)
endif()

ad_add_pybind_module(foretree src/foretree_pybind.cpp)
ad_add_nanobind_module(chomp src/chomp.cpp)

# ------------------------------------------------------------------
# Optional: SuiteSparse / CHOLMOD (for HYKKTCholmodStrategy)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Optional: SuiteSparse / CHOLMOD (for HYKKTCholmodStrategy)
# ------------------------------------------------------------------
option(AD_ENABLE_CHOLMOD "Enable HYKKT via SuiteSparse CHOLMOD" ON)

set(AD_USE_CHOLMOD OFF)
set(AD_CHOLMOD_PROVIDER "")
set(AD_CHOLMOD_INCLUDE_DIRS "")
set(AD_CHOLMOD_LIBRARIES "")

if(AD_ENABLE_CHOLMOD)
  # Try modern SuiteSparse config packages first
  find_package(SuiteSparse CONFIG)
  if(TARGET SuiteSparse::cholmod)
    set(AD_USE_CHOLMOD ON)
    set(AD_CHOLMOD_PROVIDER "SuiteSparse::cholmod")
  else()
    # Try a Cholmod config
    find_package(Cholmod) # some distros expose Cholmod::Cholmod
    if(TARGET Cholmod::Cholmod OR Cholmod_FOUND)
      set(AD_USE_CHOLMOD ON)
      set(AD_CHOLMOD_PROVIDER "Cholmod::Cholmod")
    else()
      # Fallback: pkg-config
      find_package(PkgConfi)
      if(PKG_CONFIG_FOUND)
        pkg_check_modules(CHOLMOD cholmod)
        if(CHOLMOD_FOUND)
          set(AD_USE_CHOLMOD ON)
          set(AD_CHOLMOD_INCLUDE_DIRS ${CHOLMOD_INCLUDE_DIRS})
          set(AD_CHOLMOD_LIBRARIES    ${CHOLMOD_LINK_LIBRARIES})
          set(AD_CHOLMOD_PROVIDER "pkg-config:cholmod")
        endif()
      endif()
    endif()
  endif()

  if(AD_USE_CHOLMOD)
    message(STATUS "CHOLMOD enabled via: ${AD_CHOLMOD_PROVIDER}")
  else()
    message(WARNING "AD_ENABLE_CHOLMOD=ON but CHOLMOD was not found. "
                    "HYKKT cholmod strategy will be disabled.")
  endif()
endif()

# Helper: attach CHOLMOD + EIGEN_CHOLMOD_SUPPORT to a target
function(ad_use_cholmod tgt)
include_directories(/usr/include/suitesparse)

    target_include_directories(${tgt} PRIVATE ${AD_CHOLMOD_INCLUDE_DIRS})
    target_link_libraries(${tgt} PRIVATE -lcholmod)
set(EIGEN_CHOLMOD_SUPPORT ON)

endfunction()

ad_use_cholmod(chomp)
#ad_add_pybind_module(lbl_cpp src/lbl_bindings.cpp)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
message(STATUS "==== Build summary ====")
message(STATUS "C++ Standard          : C++${CMAKE_CXX_STANDARD}")
message(STATUS "Python3               : ${Python3_EXECUTABLE} (${Python3_VERSION})")
message(STATUS "pybind11              : ${pybind11_VERSION}")
message(STATUS "Eigen3                : ${Eigen3_VERSION}")
message(STATUS "OpenMP enabled        : ${AD_ENABLE_OPENMP} (found: ${OpenMP_CXX_FOUND})")
message(STATUS "Build piqp_cpp        : ${AD_ENABLE_PIQP}")
message(STATUS "Build osqp_cpp        : ${AD_ENABLE_OSQP}")
message(STATUS "Build qdldl_cpp       : ${AD_ENABLE_QDLDL}")
message(STATUS "Link GUROBI           : ${AD_ENABLE_GUROBI} (root: ${GUROBI_ROOT})")
