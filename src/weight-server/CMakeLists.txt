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


# Add the executable.
aux_source_directory(. WEIGHT_SRCS)
add_executable(weightserver ${WEIGHT_SRCS})
target_link_libraries(weightserver PUBLIC ${ZMQ_LIB} Threads::Threads)
target_compile_options(weightserver PRIVATE "-Wall" "-Werror")


# Move the binary to 'build/' dir.
file(RENAME coordserver ${PROJECT_SOURCE_DIR}/build)
