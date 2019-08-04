cmake_minimum_required(VERSION 3.5)


# Include AWS SDK (only the lambda threads component).
find_package(AWSSDK REQUIRED COMPONENTS lambda)


# Add the executable.
aux_source_directory(. COORD_SRCS)
add_executable(coordserver ${COORD_SRCS})
target_link_libraries(coordserver PUBLIC ${ZMQ_LIB} ${AWSSDK_LINK_LIBRARIES} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(coordserver PRIVATE "-Wall" "-Werror")
