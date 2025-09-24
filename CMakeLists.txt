cmake_minimum_required(VERSION 3.20)

# ------------------------------------------------------------------
# Project
# ------------------------------------------------------------------
project(ad VERSION 0.1.0 LANGUAGES CXX C)

# C++ standard (use features without GNU extensions)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Position independent code (safe for shared/pybind modules)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build options
option(AD_ENABLE_OPENMP   "Enable OpenMP"                     OFF)
option(AD_ENABLE_PIQP     "Build piqp_cpp Python bindings"    OFF)
option(AD_ENABLE_OSQP     "Build osqp_cpp Python bindings"    OFF)
option(AD_ENABLE_QDLDL    "Build qdldl_cpp Python bindings"   OFF)
option(AD_ENABLE_AMDQG    "Build amdqg Python bindings"       OFF)
option(AD_ENABLE_L1CORE   "Build l1core Python bindings"      OFF)
option(AD_ENABLE_CHOLMOD  "Enable HYKKT via SuiteSparse"      OFF)
option(AD_ENABLE_SPECTRA  "Fetch and enable Spectra via CPM"  OFF)
option(AD_ENABLE_GUROBI   "Enable GUROBI linkage"             OFF)
option(AD_ENABLE_STDEXEC "Enable stdexec usage"               OFF)
option(AD_ENABLE_FORETREE "Build foretree Python bindings"   OFF)

# ------------------------------------------------------------------
# CPM bootstrap (before any CPMAddPackage)
# ------------------------------------------------------------------
include(cmake/CPM.cmake OPTIONAL)
if(NOT COMMAND CPMAddPackage)
  include(FetchContent)
  FetchContent_Declare(
    CPM
    GIT_REPOSITORY https://github.com/cpm-cmake/CPM.cmake.git
    GIT_TAG v0.42.0
  )
  FetchContent_MakeAvailable(CPM)
  include(${cpm_SOURCE_DIR}/cmake/CPM.cmake)
endif()

# ------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------

if (AD_ENABLE_STDEXEC)
  message(STATUS "stdexec enabled")
  cpmaddpackage(
  NAME stdexec
  GITHUB_REPOSITORY NVIDIA/stdexec
  GIT_TAG main
  OPTIONS "STDEXEC_BUILD_TESTS OFF" "STDEXEC_BUILD_EXAMPLES OFF" "STDEXEC_BUILD_BENCHMARKS OFF" "STDEXEC_BUILD_DOCS OFF" "BUILD_TESTING OFF"
)
else()
  message(STATUS "stdexec disabled")
endif()

# fmt
find_package(fmt CONFIG QUIET)
if(NOT fmt_FOUND)
  message(STATUS "fmt not found; fetching via CPM")
  CPMAddPackage(NAME fmt GITHUB_REPOSITORY fmtlib/fmt GIT_TAG 11.2.0)
endif()

# pybind11
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found; fetching via CPM")
  CPMAddPackage(NAME pybind11 GITHUB_REPOSITORY pybind/pybind11 GIT_TAG v2.13.6)
endif()

# Eigen3
find_package(Eigen3 3.3 CONFIG QUIET)
if(NOT Eigen3_FOUND)
  message(STATUS "Eigen3 not found; fetching via CPM")
  CPMAddPackage(
    NAME Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
  )
  # Some CPM recipes export 'eigen' target; create alias if needed.
  if(TARGET eigen)
    add_library(Eigen3::Eigen ALIAS eigen)
  elseif(NOT TARGET Eigen3::Eigen)
    # Header-only fallback
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set(_eigen_inc "${Eigen3_SOURCE_DIR}")
    if(EXISTS "${Eigen3_SOURCE_DIR}/Eigen")
      set(_eigen_inc "${Eigen3_SOURCE_DIR}")
    elseif(EXISTS "${Eigen3_SOURCE_DIR}/include/eigen3")
      set(_eigen_inc "${Eigen3_SOURCE_DIR}/include/eigen3")
    endif()
    set_target_properties(Eigen3::Eigen PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${_eigen_inc}"
    )
  endif()
endif()

# Python 3 — Interpreter + Development.Module (for nanobind) + Development
find_package(Python3 3.13
  COMPONENTS Interpreter Development Development.Module
  REQUIRED
)

# define include dir python globally
include_directories(${Python3_INCLUDE_DIRS})

# nanobind (always via CPM is fine; it uses Python3 variables)
# nanobind (always via CPM is fine; it uses Python3 variables)
set(NANOBIND_TAG v2.9.2 CACHE STRING "nanobind version/tag to fetch via CPM")
CPMAddPackage(
  NAME nanobind
  GITHUB_REPOSITORY wjakob/nanobind
  GIT_TAG ${NANOBIND_TAG}
  OPTIONS "NB_STATIC=ON"
)



# Spectra (header-only; CPM + INTERFACE target alias)
if(AD_ENABLE_SPECTRA)
  CPMAddPackage(
    NAME Spectra
    GITHUB_REPOSITORY yixuan/spectra
    GIT_TAG v1.2.0
    DOWNLOAD_ONLY NO
  )
  # Ensure an imported/alias target exists: Spectra::Spectra
  if(NOT TARGET Spectra::Spectra)
    if(TARGET Spectra) # some recipes create 'Spectra'
      add_library(Spectra::Spectra INTERFACE IMPORTED)
      target_link_libraries(Spectra::Spectra INTERFACE Spectra)
    else()
      # Header-only include dir fallback
      add_library(Spectra::Spectra INTERFACE IMPORTED)
      # Typical include path is <pkg>/include
      if(DEFINED Spectra_SOURCE_DIR)
        target_include_directories(Spectra::Spectra INTERFACE
          "${Spectra_SOURCE_DIR}/include")
      elseif(DEFINED Spectra_INCLUDE_DIRS)
        target_include_directories(Spectra::Spectra INTERFACE
          "${Spectra_INCLUDE_DIRS}")
      endif()
    endif()
  endif()
endif()

# OpenMP (optional)
if(AD_ENABLE_OPENMP)
  find_package(OpenMP QUIET)
endif()

# ------------------------------------------------------------------
# Optional: GUROBI
# ------------------------------------------------------------------
set(GUROBI_ROOT "" CACHE PATH "Path to GUROBI linux64 installation root")
function(_ad_link_gurobi tgt)
  if(NOT AD_ENABLE_GUROBI)
    return()
  endif()

  if(NOT GUROBI_ROOT)
    set(GUROBI_ROOT "/home/seman/gurobi1101/linux64" CACHE PATH "" FORCE)
  endif()

  set(_gurobi_inc "${GUROBI_ROOT}/include")
  set(_gurobi_lib "${GUROBI_ROOT}/lib")

  if(EXISTS "${_gurobi_inc}" AND EXISTS "${_gurobi_lib}")
    target_include_directories(${tgt} PRIVATE "${_gurobi_inc}")
    find_library(GUROBI_SHARED NAMES gurobi gurobi110 PATHS "${_gurobi_lib}")
    find_library(GUROBI_CPPA  NAMES gurobi_c++            PATHS "${_gurobi_lib}")
    if(GUROBI_SHARED AND GUROBI_CPPA)
      target_link_libraries(${tgt} PRIVATE "${GUROBI_CPPA}" "${GUROBI_SHARED}")
      if(UNIX AND NOT APPLE)
        set_property(TARGET ${tgt} APPEND PROPERTY BUILD_RPATH  "${_gurobi_lib}")
        set_property(TARGET ${tgt} APPEND PROPERTY INSTALL_RPATH "${_gurobi_lib}")
      endif()
    else()
      message(WARNING "GUROBI libs not found under ${_gurobi_lib}")
    endif()
  else()
    message(WARNING "GUROBI not found at GUROBI_ROOT='${GUROBI_ROOT}'. "
                    "To disable, set -DAD_ENABLE_GUROBI=OFF.")
  endif()
endfunction()

# ------------------------------------------------------------------
# Optional: SuiteSparse / CHOLMOD (for HYKKTCholmodStrategy)
# ------------------------------------------------------------------
set(AD_USE_CHOLMOD OFF)
set(AD_CHOLMOD_PROVIDER "")
if(AD_ENABLE_CHOLMOD)
  # Prefer modern package configs
  find_package(SuiteSparse CONFIG QUIET)
  if(TARGET SuiteSparse::cholmod)
    set(AD_USE_CHOLMOD ON)
    set(AD_CHOLMOD_PROVIDER "SuiteSparse::cholmod")
  else()
    find_package(Cholmod QUIET) # some distros export Cholmod::Cholmod
    if(TARGET Cholmod::Cholmod)
      set(AD_USE_CHOLMOD ON)
      set(AD_CHOLMOD_PROVIDER "Cholmod::Cholmod")
    else()
      find_package(PkgConfig QUIET)
      if(PKG_CONFIG_FOUND)
        pkg_check_modules(CHOLMOD QUIET cholmod)
        if(CHOLMOD_FOUND)
          set(AD_USE_CHOLMOD ON)
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

function(ad_use_cholmod tgt)
  if(NOT AD_USE_CHOLMOD)
    return()
  endif()
  if(AD_CHOLMOD_PROVIDER STREQUAL "SuiteSparse::cholmod")
    target_link_libraries(${tgt} PRIVATE SuiteSparse::cholmod)
    target_compile_definitions(${tgt} PRIVATE EIGEN_CHOLMOD_SUPPORT=1)
  elseif(AD_CHOLMOD_PROVIDER STREQUAL "Cholmod::Cholmod")
    target_link_libraries(${tgt} PRIVATE Cholmod::Cholmod)
    target_compile_definitions(${tgt} PRIVATE EIGEN_CHOLMOD_SUPPORT=1)
  else()
    # pkg-config fallback
    target_include_directories(${tgt} PRIVATE ${CHOLMOD_INCLUDE_DIRS})
    target_link_libraries(${tgt} PRIVATE ${CHOLMOD_LINK_LIBRARIES})
    target_compile_definitions(${tgt} PRIVATE EIGEN_CHOLMOD_SUPPORT=1)
  endif()
endfunction()

# ------------------------------------------------------------------
# Helpers to create Python extension modules
# ------------------------------------------------------------------
function(ad__common_target_setup tgt)
  target_compile_features(${tgt} PRIVATE cxx_std_23)
  target_include_directories(${tgt}
    PRIVATE
      "${PROJECT_SOURCE_DIR}/src"
  )
  target_link_libraries(${tgt}
    PRIVATE
      fmt::fmt
      Eigen3::Eigen
  )
  if(AD_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  if(AD_ENABLE_SPECTRA AND TARGET Spectra::Spectra)
    target_link_libraries(${tgt} PRIVATE Spectra::Spectra)
    target_compile_definitions(${tgt} PRIVATE AD_HAVE_SPECTRA=1)
  endif()
  if(MSVC)
    target_compile_options(${tgt} PRIVATE /W4)
  else()
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic -fvisibility=hidden)
  endif()
endfunction()

function(ad_add_pybind_module name)
  # Usage: ad_add_pybind_module(<name> src1.cpp src2.cpp ...)
  set(_srcs ${ARGN})
  pybind11_add_module(${name} ${_srcs})
  message(STATUS "Added pybind11 module: ${name} with sources: ${_srcs}")
  ad__common_target_setup(${name})
endfunction()

function(ad_add_nanobind_module name)
  # Usage: ad_add_nanobind_module(<name> src1.cpp src2.cpp ...)
  set(_srcs ${ARGN})
  nanobind_add_module(${name} ${_srcs})
  message(STATUS "Added nanobind module: ${name} with sources: ${_srcs}")
  ad__common_target_setup(${name})
  # nanobind modules need Python
  target_link_libraries(${name} PRIVATE Python3::Module)
endfunction()

# ------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------

# Core AD Python module
set(AD_SOURCES
  src/ad_bindings.cpp
  src/ad/Variable.cpp
  src/ad/Expression.cpp
  src/ad/ADGraph.cpp
)
ad_add_nanobind_module(ad ${AD_SOURCES})
target_link_libraries(ad PRIVATE m dl)  # system libs where available
# _ad_link_gurobi(ad)                      # optional GUROBI

# chomp (nanobind)
ad_add_nanobind_module(chomp
  src/chomp.cpp
  src/ad/ADGraph.cpp
  src/ad/Expression.cpp
  src/ad/Variable.cpp
)
ad_use_cholmod(chomp)
# add python dir to ad module search path
target_include_directories(chomp PRIVATE ${Python3_INCLUDE_DIRS})
target_include_directories(ad PRIVATE ${Python3_INCLUDE_DIRS})

# _ad_link_gurobi(chomp)
# link openmp to chomp if available

  find_package(TBB COMPONENTS tbb)
if(TARGET TBB::tbb)
  target_link_libraries(chomp PRIVATE TBB::tbb)
else()
  message(WARNING "oneTBB not found. Building foreforest without parallel backend.")
  target_compile_definitions(chomp PRIVATE FF_NO_TBB)
endif()

if(TARGET TBB::tbb)
  target_link_libraries(ad PRIVATE TBB::tbb)
else()
  message(WARNING "oneTBB not found. Building foreforest without parallel backend.")
  target_compile_definitions(ad PRIVATE FF_NO_TBB)
endif()

if(AD_ENABLE_FORETREE)
# foretree (pybind11)
ad_add_pybind_module(foretree src/foretree_pybind.cpp)
target_include_directories(foreforest PRIVATE ${stdexec_SOURCE_DIR}/include)
find_package(TBB QUIET COMPONENTS tbb)
if(TARGET TBB::tbb)
  target_link_libraries(foreforest PRIVATE TBB::tbb)
else()
  message(WARNING "oneTBB not found. Building foreforest without parallel backend.")
  target_compile_definitions(foreforest PRIVATE FF_NO_TBB)
endif()
endif()


# Optional modules
if(AD_ENABLE_L1CORE)
  ad_add_pybind_module(l1core
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
  ad_add_pybind_module(simplex_core src/simplex_bindings.cpp)
endif()

if(AD_ENABLE_PIQP)
  ad_add_pybind_module(piqp_cpp src/piqp_bindings.cpp)
endif()

if(AD_ENABLE_AMDQG)
  ad_add_pybind_module(amdqg src/amdqg_bindings.cpp)
endif()

if(AD_ENABLE_OSQP)
  ad_add_pybind_module(osqp_cpp src/osqp_bindings.cpp)
endif()

if(AD_ENABLE_QDLDL)
  ad_add_pybind_module(qdldl_cpp src/qdldl_bindings.cpp)
endif()

# target_include_directories(ad PRIVATE ${stdexec_SOURCE_DIR}/include)
# target_include_directories(chomp PRIVATE ${stdexec_SOURCE_DIR}/include)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
message(STATUS "==== Build summary ====")
message(STATUS "C++ Standard          : C++${CMAKE_CXX_STANDARD}")
message(STATUS "Python3               : ${Python3_EXECUTABLE} (${Python3_VERSION})")
if(DEFINED pybind11_VERSION)
  message(STATUS "pybind11              : ${pybind11_VERSION}")
endif()
if(DEFINED Eigen3_VERSION)
  message(STATUS "Eigen3                : ${Eigen3_VERSION}")
endif()
message(STATUS "OpenMP enabled        : ${AD_ENABLE_OPENMP} (found: ${OpenMP_CXX_FOUND})")
message(STATUS "Enable Spectra        : ${AD_ENABLE_SPECTRA}")
message(STATUS "Build piqp_cpp        : ${AD_ENABLE_PIQP}")
message(STATUS "Build osqp_cpp        : ${AD_ENABLE_OSQP}")
message(STATUS "Build qdldl_cpp       : ${AD_ENABLE_QDLDL}")
message(STATUS "Link GUROBI           : ${AD_ENABLE_GUROBI} (root: ${GUROBI_ROOT})")
message(STATUS "CHOLMOD enabled       : ${AD_ENABLE_CHOLMOD} (use: ${AD_USE_CHOLMOD})")
