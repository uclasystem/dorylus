cmake_minimum_required(VERSION 3.5)


# Add the library objects.
add_library(edge "edge.cpp")
target_link_libraries(edge PRIVATE utils
                           PUBLIC ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(edge PRIVATE "-Wall" "-Werror" "-Wno-sign-compare" "-Wno-reorder" "-MMD")
add_library(vertex "vertex.cpp")
target_link_libraries(vertex PRIVATE utils
                             PUBLIC ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(vertex PRIVATE "-Wall" "-Werror" "-Wno-sign-compare" "-Wno-reorder" "-MMD")
add_library(graph "graph.cpp")
target_link_libraries(graph PRIVATE utils
                            PUBLIC ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(graph PRIVATE "-Wall" "-Werror" "-Wno-sign-compare" "-Wno-reorder" "-MMD")
add_library(engine "engine.cpp")
target_link_libraries(engine PRIVATE edge vertex graph commmanager resource_comm nodemanager threadpool utils
                             PUBLIC  ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(engine PRIVATE "-Wall" "-Werror" "-Wno-sign-compare" "-Wno-reorder" "-MMD")
