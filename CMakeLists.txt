cmake_minimum_required(VERSION 3.5)


# Options.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)


# Project name.
project(gnn-lambda LANGUAGES CXX)


# Include ZeroMQ.
set(ZMQ_INC /usr/local/include)
set(ZMQ_LIB /usr/local/lib/libzmq.a)
include_directories(${ZMQ_INC})


# Include ZooKeeper
set(ZK_INC /usr/local/include/zookeeper)
set(ZK_LIB /usr/local/lib/libzookeeper_mt.a)
include_directories(${ZK_INC})


# Include Boost lib.
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})


# Include Threads lib.
find_package(Threads REQUIRED)


# Graph server (Data server).
add_subdirectory(src/graph-server ${PROJECT_SOURCE_DIR}/build/graph-server)


# Coordination server.
add_subdirectory(src/coord-server ${PROJECT_SOURCE_DIR}/build/coord-server)


# Weight server.
add_subdirectory(src/weight-server ${PROJECT_SOURCE_DIR}/build/weight-server)
