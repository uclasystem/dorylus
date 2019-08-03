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


# Include AWS SDK (only the lambda threads component).
find_package(AWSSDK REQUIRED COMPONENTS lambda)


# Add the executable.
aux_source_directory(. COORD_SRCS)
add_executable(coordserver ${COORD_SRCS})
target_link_libraries(coordserver PUBLIC ${ZMQ_LIB} ${AWSSDK_LINK_LIBRARIES} Threads::Threads)
target_compile_options(coordserver PRIVATE "-Wall" "-Werror")


# Move the binary to 'build/' dir.
file(RENAME coordserver ${PROJECT_SOURCE_DIR}/build)
