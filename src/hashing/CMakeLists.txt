add_executable(hashing_main main.cpp)

add_library(hashing INTERFACE HashMap.h)
target_include_directories(hashing INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(string_hashing StringHashFunctions.cpp StringHashFunctions.h)
