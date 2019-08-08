cmake_minimum_required(VERSION 3.5)


# Add the library objects.
add_library(threadpool "threadpool.cpp")
target_link_libraries(threadpool PUBLIC ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(threadpool PRIVATE "-Wall" "-Werror" "-MMD")
