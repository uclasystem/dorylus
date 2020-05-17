cmake_minimum_required(VERSION 3.5)


aux_source_directory(. OP_SRC)
add_library(ops SHARED ${OP_SRC})
target_link_libraries(ops PUBLIC ${OBLIB} ${CBLIB})
set_property(TARGET ops PROPERTY POSITION_INDEPENDENT_CODE ON)
