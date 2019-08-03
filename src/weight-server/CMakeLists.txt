cmake_minimum_required(VERSION 3.5)


# Options.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)


# Include ZeroMQ.
set(INSTALL_OUT_PATH ${PROJECT_SOURCE_DIR}/installs/out)
set(ZMQ_INC ${INSTALL_OUT_PATH}/include)
set(ZMQ_LIB ${INSTALL_OUT_PATH}/lib/libzmq.a)
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
