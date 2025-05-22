cmake_minimum_required(VERSION 3.10)
project(cactus_vlm)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-g -O0 -fsanitize=address -fno-omit-frame-pointer)
    link_libraries(-fsanitize=address)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(../../cactus ${CMAKE_BINARY_DIR}/cactus_core_build)

# Executable for the FFI example
add_executable(cactus_vlm_ffi 
    main_ffi.cpp
)

target_link_libraries(cactus_vlm_ffi
    PRIVATE
    cactus_core_lib
)

# Executable for the Core API example
add_executable(cactus_vlm_core
    main_core.cpp
)

target_link_libraries(cactus_vlm_core
    PRIVATE
    cactus_core_lib
)