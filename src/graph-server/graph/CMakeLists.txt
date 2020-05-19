cmake_minimum_required(VERSION 3.5)


# Add the library objects.
add_library(graph "graph.cpp" "vertex.cpp" "edge.cpp" "dataloader.cpp")
target_link_libraries(graph PRIVATE utils
                            PUBLIC ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(graph PRIVATE "-Wall" "-Werror" "-Wno-sign-compare" "-Wno-reorder" "-MMD")
