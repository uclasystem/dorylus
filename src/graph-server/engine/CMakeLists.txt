cmake_minimum_required(VERSION 3.5)


# Add the library objects.
add_library(edge "edge.cpp")
target_link_libraries(edge PUBLIC ${ZMQ_LIB} ${ZK_LIB} Threads::Threads ${BOOST_LIBRARIES})
target_compile_options(edge PRIVATE "-Wall" "-Werror" "-Wno-sign-compare" "-Wno-reorder" "-MMD")
add_library(vertex "vertex.cpp")
target_link_libraries(vertex PUBLIC ${ZMQ_LIB} ${ZK_LIB} Threads::Threads ${BOOST_LIBRARIES})
target_compile_options(vertex PRIVATE "-Wall" "-Werror" "-Wno-sign-compare" "-Wno-reorder" "-MMD")
add_library(graph "graph.cpp")
target_link_libraries(graph PUBLIC ${ZMQ_LIB} ${ZK_LIB} Threads::Threads ${BOOST_LIBRARIES})
target_compile_options(graph PRIVATE "-Wall" "-Werror" "-Wno-sign-compare" "-Wno-reorder" "-MMD")
add_library(engine "engine.cpp")
target_link_libraries(engine PUBLIC ${ZMQ_LIB} ${ZK_LIB} Threads::Threads ${BOOST_LIBRARIES})
target_compile_options(engine PRIVATE "-Wall" "-Werror" "-Wno-sign-compare" "-Wno-reorder" "-MMD")
