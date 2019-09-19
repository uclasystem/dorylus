cmake_minimum_required(VERSION 3.5)


# Add the library objects.
add_library(utils "utils.cpp")
set_property(TARGET utils PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(utils PUBLIC ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(utils PRIVATE "-Wall" "-Werror" "-MMD")
