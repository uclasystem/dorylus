cmake_minimum_required(VERSION 3.5)


# Add the library objects.
add_library(nodemanager "nodemanager.cpp")
target_link_libraries(nodemanager PUBLIC ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(nodemanager PRIVATE "-Wall" "-Werror" "-MMD")
