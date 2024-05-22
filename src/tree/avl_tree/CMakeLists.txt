add_executable(avl_main main.cpp)
set_target_properties(avl_main PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/executables")

add_library(avl_tree INTERFACE AVLTree.h)
target_include_directories(avl_tree INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(avl_tree PRIVATE INTERFACE bst)
target_link_libraries(avl_main avl_tree)
