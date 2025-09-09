cmake_minimum_required(VERSION 3.10)  # Set the minimum version of CMake

# Set your project name and version
project(OptimizationProject VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
#add_compile_options(-g -Wall -Wextra)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Enable Link Time Optimization (LTO)
# include(CheckIPOSupported)
# check_ipo_supported(RESULT result)
# if(result)
#     set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
# endif()

# Optimize for the native architecture
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -flto=auto")

# Optionally, set parallel LTO jobs
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=auto")

# Attempt to include CPM.cmake from provided path
include(cmake/CPM.cmake OPTIONAL)

# Check if CPM was included, if not, fetch and include CPM
if (NOT COMMAND CPMAddPackage)
    # Include FetchContent module
    include(FetchContent)
    # Declare CPM.cmake as a FetchContent
    FetchContent_Declare(
            CPM
            GIT_REPOSITORY https://github.com/cpm-cmake/CPM.cmake.git
            GIT_TAG v0.38.7 # You can specify a specific version of CPM.cmake
    )
    # Fetch CPM
    FetchContent_MakeAvailable(CPM)
    # Include CPM.cmake after it has been fetched
    include(${cpm_SOURCE_DIR}/cmake/CPM.cmake)
endif()

CPMAddPackage(
        NAME fmt
        GITHUB_REPOSITORY fmtlib/fmt
        GIT_TAG 10.2.1 # Specify the version of fmt you want to use
)

# Eigen setup
CPMAddPackage(
  NAME Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG master # Specify the version of Eigen you want to use
)

# Add Pybind11 via CPM
CPMAddPackage(
  NAME pybind11
  GITHUB_REPOSITORY pybind/pybind11
  VERSION 2.12.0
)

# Add the library
add_library(l1penalty SHARED
        src/main.cpp
        src/Solvers.cpp
        src/PolynomialVector.cpp
        src/pointWork.cpp
        src/Overloads.cpp
        src/l1Solver.cpp
        src/Helpers.cpp
        src/l1Fun.cpp
        src/TRModel.cpp
)

# declare public API
# set_target_properties(l1penalty PROPERTIES PUBLIC_HEADER include/Definitions.hpp include/l1Solver.cpp)
set_property(TARGET l1penalty PROPERTY
        PUBLIC_HEADER
        include/Definitions.hpp
        include/l1Solver.hpp 
        include/CPublic.hpp
)


target_include_directories(l1penalty
                           INTERFACE
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                            $<INSTALL_INTERFACE:include>
                           )
target_include_directories(l1penalty PUBLIC "$ENV{GUROBI_HOME}/include")

target_link_libraries(l1penalty
        PUBLIC "$ENV{GUROBI_HOME}/lib/libgurobi_c++.a"
        PUBLIC "$ENV{GUROBI_HOME}/lib/libgurobi120.so"
        PUBLIC fmt::fmt
        PUBLIC Eigen3::Eigen
)


# Add the executable
add_executable(solve src/main.cpp
        src/Solvers.cpp
        src/PolynomialVector.cpp
        src/pointWork.cpp
        src/Overloads.cpp
        src/l1Solver.cpp
        src/Helpers.cpp
        src/l1Fun.cpp
        src/TRModel.cpp
)

target_include_directories(solve PUBLIC "$ENV{GUROBI_HOME}/include")



include_directories(${Eigen_SOURCE_DIR})

# link fmt to your executable
target_link_libraries(solve
        PUBLIC "$ENV{GUROBI_HOME}/lib/libgurobi_c++.a"
        PUBLIC "$ENV{GUROBI_HOME}/lib/libgurobi120.so"
        PUBLIC fmt::fmt
        PUBLIC Eigen3::Eigen
)

pybind11_add_module(l1 src/bindings.cpp
        src/Solvers.cpp
        src/PolynomialVector.cpp
        src/pointWork.cpp
        src/Overloads.cpp
        src/l1Solver.cpp
        src/Helpers.cpp
        src/l1Fun.cpp
        src/TRModel.cpp
)

target_link_libraries(l1 PRIVATE pybind11::module)
target_include_directories(l1 PUBLIC "$ENV{GUROBI_HOME}/include")
target_link_libraries(l1 PRIVATE "$ENV{GUROBI_HOME}/lib/libgurobi_c++.a"
        "$ENV{GUROBI_HOME}/lib/libgurobi120.so")
        target_link_libraries(l1 PRIVATE fmt::fmt)
        target_link_libraries(l1 PRIVATE Eigen3::Eigen)
        
# If Eigen is installed system-wide, find it
#find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Link Eigen to your executabless