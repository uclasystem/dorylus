cmake_minimum_required(VERSION 3.5)


# Options.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DVERBOSE_ERRORS -fno-omit-frame-pointer -fsanitize=address")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()


# Project name.
project(gnn-lambda LANGUAGES CXX)


# Include ZeroMQ.
set(ZMQ_INC /usr/include)
set(ZMQ_LIB /usr/lib/libzmq.so)
include_directories(${ZMQ_INC})


# Include Boost lib.
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})


# Include Threads lib.
find_package(Threads REQUIRED)


# Graph server (Data server).
if(BUILD_GRAPH)
    message("Graph server:  YES")
    add_subdirectory(src/graph-server ${PROJECT_SOURCE_DIR}/build/graph-server)
else()
    message("Graph server:  NO")
endif()


# Coordination server.
if(BUILD_COORD)
    message("Coord server:  YES")
    add_subdirectory(src/coord-server ${PROJECT_SOURCE_DIR}/build/coord-server)
else()
    message("Coord server:  NO")
endif()


# Weight server.
if(BUILD_WEIGHT)
    message("Weight server: YES")
    add_subdirectory(src/weight-server ${PROJECT_SOURCE_DIR}/build/weight-server)
else()
    message("Weight server: NO")
endif()
