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


# Add the executable.
aux_source_directory(. WEIGHT_SRCS)
add_executable(weightserver ${WEIGHT_SRCS})
target_link_libraries(weightserver PUBLIC ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(weightserver PRIVATE "-Wall" "-Werror")
