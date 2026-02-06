cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(varintrvv VERSION 0.1 LANGUAGES C CXX ASM)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build.")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Do not build shared libraries")
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(BUILD_TESTING OFF CACHE BOOL "Disable all tests" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable Benchmark's tests" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable Benchmark's GTest tests")
set(BUILD_EXAMPLE CACHE BOOL OFF)
set(BUILD_BENCHMARK CACHE BOOL OFF)

set(ARCH "-march=rv64gcv")
add_compile_options(${ARCH} -O3)

add_subdirectory(submodules/google-benchmark)

# Add include directories
include_directories(${PROJECT_SOURCE_DIR}/lib/include)

# Specify source files
set(SOURCE_FILES 
    ${PROJECT_SOURCE_DIR}/lib/src/varint_encode.c
    ${PROJECT_SOURCE_DIR}/lib/src/varint_decode_scalar.c
    # ${PROJECT_SOURCE_DIR}/lib/src/varint_decode_vecshift.c
    ${PROJECT_SOURCE_DIR}/lib/src/varint_rvv.S
    )

add_library(varintrvv STATIC ${SOURCE_FILES})

#build example executable
if(BUILD_EXAMPLE)
    add_executable(example ${PROJECT_SOURCE_DIR}/example/example.c)
    target_link_libraries(example varintrvv -static)
endif()

if(BUILD_BENCHMARK)
    add_executable(varint_benchmark ${PROJECT_SOURCE_DIR}/benchmark/benchmark.cc)
    target_link_libraries(varint_benchmark PRIVATE benchmark::benchmark varintrvv pthread -static)
endif()
