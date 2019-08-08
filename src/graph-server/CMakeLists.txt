cmake_minimum_required(VERSION 3.5)


# Subdirectories, each needs a CMakeLists.txt.
add_subdirectory(commmanager)
add_subdirectory(nodemanager)
add_subdirectory(engine)
add_subdirectory(parallel)
add_subdirectory(utils)


# Add the executable.
aux_source_directory(. GRAPH_SRCS)
add_executable(graphserver ${GRAPH_SRCS})
target_link_libraries(graphserver PRIVATE commmanager lambda_comm edge vertex graph engine nodemanager threadpool utils
                                  PUBLIC  ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(graphserver PRIVATE "-Wall" "-Werror" "-Wno-reorder" "-MMD")
