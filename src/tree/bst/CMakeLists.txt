add_executable(bst_main main.cpp)
set_target_properties(bst_main PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/executables")

add_library(bst INTERFACE BST.h)
target_include_directories(bst INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(bst_main bst)
