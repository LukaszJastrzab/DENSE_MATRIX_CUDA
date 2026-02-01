cmake_minimum_required(VERSION 3.19)
project(DenseMatrixCuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include( FetchContent )

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable( googletest )

enable_testing()

find_package(CUDAToolkit REQUIRED)

add_executable(${PROJECT_NAME}
    tests/main.cu
    tests/functions.cuh
    src/dense_matrix_cuda.cuh
    src/dense_matrix_cuda.cu
    external/dense_matrix.hpp
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G -lineinfo>
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 75;86;89)

target_link_libraries(${PROJECT_NAME} PRIVATE
    GTest::gtest
    GTest::gtest_main
    CUDA::cudart
)

if (MSVC)
    add_compile_definitions(__CUDACC__)
endif()

target_include_directories( ${PROJECT_NAME}
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/external
    ${CUDAToolkit_INCLUDE_DIRS}
)

include( GoogleTest )
gtest_discover_tests( ${PROJECT_NAME} )