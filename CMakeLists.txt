cmake_minimum_required(VERSION 3.5)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

project(ad LANGUAGES C CXX)

# Set C++17 as the standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the C and C++ compilers to musl-gcc set(CMAKE_C_COMPILER "gcc")
# set(CMAKE_CXX_COMPILER "g++") include(CPM.cmake)

# Attempt to include CPM.cmake from provided path
include(cmake/CPM.cmake OPTIONAL)

# Check if CPM was included, if not, fetch and include CPM
if(NOT COMMAND CPMAddPackage)
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

cpmaddpackage(
  NAME fmt GITHUB_REPOSITORY fmtlib/fmt GIT_TAG 11.0.1 # Specify the version of
                                                       # fmt you want to use
)

# Include directories
include_directories(/usr/include/python3.12)
# include_directories(/data/sqp_solver/src)

include_directories(/usr/include/eigen3/)
# include_directories(/usr/include/suitesparse/)
# include_directories(/usr/lib/python3.12/site-packages/numpy/core/include)
include_directories(/home/seman/gurobi1101/linux64/include)
# include_directories(/home/knitro/gurobi1100/linux64/include)

# Find Python, pybind11, and OpenMP
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
find_package(OpenMP REQUIRED)
# find_package(Python3 3.12 COMPONENTS Interpreter Development REQUIRED)

# find_package(CUDA REQUIRED)
message(STATUS "Python Executable: ${Python3_EXECUTABLE}")
message(STATUS "Python Version: ${Python3_VERSION}")
message(STATUS "Python Include Dir: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python Libraries: ${Python3_LIBRARIES}")
# Add the executable
add_library(
  ad MODULE
  src/ad.cpp
  src/Variable.cpp
  src/Expression.cpp
  src/ADGraph.cpp)

# Link libraries
target_link_libraries(
  ad
  PRIVATE pybind11::pybind11
          -L/usr/lib64
          -L/usr/lib64/x86_64-linux-gnu
          -lcholmod
          -lm
          -ldl
          -lgomp
          /home/seman/gurobi1101/linux64/lib/libgurobi_c++.a
          /home/seman/gurobi1101/linux64/lib/libgurobi110.so)

target_link_libraries(ad PRIVATE fmt::fmt)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)


    # Create Python module
    pybind11_add_module(piqp_cpp src/piqp_bindings.cpp)
    
    # Link with Eigen
    target_link_libraries(piqp_cpp PRIVATE Eigen3::Eigen)
    
    # Compiler-specific flags for the Python module
    target_compile_definitions(piqp_cpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
    
    if(OpenMP_CXX_FOUND)
        target_link_libraries(piqp_cpp PRIVATE OpenMP::OpenMP_CXX)
    endif()



# Link OpenMP if found
if(OpenMP_CXX_FOUND)
  target_link_libraries(ad PRIVATE OpenMP::OpenMP_CXX)
endif()

# Get the Python extension suffix for Python 3.12
execute_process(
  COMMAND /usr/bin/python-config --extension-suffix
  OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# Set output file name with Python extension suffix
set_target_properties(ad PROPERTIES PREFIX "" SUFFIX
                                               ${PYTHON_EXTENSION_SUFFIX})
