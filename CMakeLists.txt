cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(varintrvv VERSION 0.1 LANGUAGES C CXX ASM)


# cmake .. -DCMAKE_C_COMPILER=riscv64-unknown-linux-gnu-gcc      -DCMAKE_CXX_COMPILER=riscv64-unknown-linux-gnu-g++ -DBENCHMARK_ENABLE_WERROR=0      -DCMAKE_SYSTEM_NAME=Generic   -DBUILD_BENCHMARK=1   -DCMAKE_SYSTEM_PROCESSOR=riscv && make -j 8

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Do not build shared libraries")
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_TESTING OFF CACHE BOOL "Disable all tests" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable Benchmark's tests" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable Benchmark's GTest tests")
set(BUILD_EXAMPLE CACHE BOOL OFF)
set(BUILD_BENCHMARK CACHE BOOL OFF)

# ADD_COMPILE_OPTIONS(-march=rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm_zfh_zfhmin_zca_zcd_zba_zbb_zbc_zbs_zkt_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin_zvkt_sscofpmf_sstc_svinval_svnapot_svpbmt -mabi=lp64d -O3 -static)
ADD_COMPILE_OPTIONS(-march=rv64gcv_zba -mabi=lp64d -O3 -static)

add_subdirectory(submodules/google-benchmark)

# set(CMAKE_C_COMPILER   riscv64-unknown-linux-gnu-gcc)
# set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-g++)

set(CMAKE_SYSROOT "/home/marco/development/riscv/toolchain/install/64-bit/sysroot")
set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)  # find host tools (pkg-config, etc.)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)   # find target libs
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)   # find target headers
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)   # find target packages

# Add include directories
include_directories(${PROJECT_SOURCE_DIR}/lib/include)

# Specify source files
set(SOURCE_FILES 
    ${PROJECT_SOURCE_DIR}/lib/src/varint_decode_maskshift.c 
    ${PROJECT_SOURCE_DIR}/lib/src/varint_encode.c
    ${PROJECT_SOURCE_DIR}/lib/src/varint_decode_scalar.c
    ${PROJECT_SOURCE_DIR}/lib/src/varint_decode_maskedvbyte.c
    ${PROJECT_SOURCE_DIR}/lib/src/varint_decode_vecshift.c
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
