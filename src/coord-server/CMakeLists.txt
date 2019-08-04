cmake_minimum_required(VERSION 3.5)


# Options.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)


# Include ZeroMQ.
set(ZMQ_INC /usr/local/include)
set(ZMQ_LIB /usr/local/lib/libzmq.a)
include_directories(${ZMQ_INC})


# Include Threads lib.
find_package(Threads REQUIRED)


# Include Boost lib.
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})


# Include AWS SDK (only the lambda threads component).
find_package(AWSSDK REQUIRED COMPONENTS lambda)


# Add the executable.
aux_source_directory(. COORD_SRCS)
add_executable(coordserver ${COORD_SRCS})
target_link_libraries(coordserver PUBLIC ${ZMQ_LIB} ${AWSSDK_LINK_LIBRARIES} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(coordserver PRIVATE "-Wall" "-Werror")
