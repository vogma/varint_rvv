cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(varintrvv VERSION 0.1 LANGUAGES C CXX)

set(BUILD_EXAMPLE CACHE BOOL OFF)

ADD_COMPILE_OPTIONS(-march=rv64gcv -mabi=lp64d -O3 -static -Werror -Wextra -g)

# Add include directories
include_directories(${PROJECT_SOURCE_DIR}/lib/include)

# Specify source files
set(SOURCE_FILES 
    ${PROJECT_SOURCE_DIR}/lib/src/varintdecode.c 
    )

# Define the static library to be built with the specified source files
add_library(varintrvv STATIC ${SOURCE_FILES})

#build example executable
if(BUILD_EXAMPLE)
    add_executable(example ${PROJECT_SOURCE_DIR}/example/compress.c)
    target_link_libraries(example varintrvv -static)
endif()
