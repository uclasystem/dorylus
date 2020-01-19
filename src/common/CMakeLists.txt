cmake_minimum_required(VERSION 3.5)


aux_source_directory(. COMMON_SRC)
add_library(common SHARED ${COMMON_SRC})
target_link_libraries(common PUBLIC ${OBLIB} ${CBLIB})
set_property(TARGET common PROPERTY POSITION_INDEPENDENT_CODE ON)
