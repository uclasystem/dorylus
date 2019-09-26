cmake_minimum_required(VERSION 3.5)

add_library(matrix SHARED "matrix.cpp")
target_link_libraries(matrix PUBLIC ${OBLIB} ${CBLIB})
set_property(TARGET matrix PROPERTY POSITION_INDEPENDENT_CODE ON)
