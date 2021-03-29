cmake_minimum_required(VERSION 3.5)


# Options.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DVERBOSE_ERRORS -fno-omit-frame-pointer -fsanitize=address -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()


# Project name.
project(dorylus LANGUAGES CXX)


# Include ZeroMQ.
find_library(ZMQ_LIB NAME libzmq.so HINTS /usr/lib/ /usr/local/lib/ REQUIRED)
set(ZMQ_INC /usr/local/include)
include_directories(${ZMQ_INC})

# Include BLAS/CBLAS
set(OBLIB /usr/local/blas/OpenBLAS/lib/libopenblas.a)
set(CBLIB /usr/local/blas/CBLAS/lib/cblas_LINUX.a)
set(CBINC /usr/local/blas/CBLAS/include)
include_directories(${CBINC})


# Include Boost lib.
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})


# Include Threads lib.
find_package(Threads REQUIRED)


# Graph server (Data server).
if(BUILD_GRAPH)
    message("Graph server:  YES")
    add_subdirectory(src/common)
    add_subdirectory(src/graph-server ${PROJECT_SOURCE_DIR}/build/graph-server)
else()
    message("Graph server:  NO")
endif()



# Weight server.
if(BUILD_WEIGHT)
    message("Weight server: YES")
    add_subdirectory(src/common)
    add_subdirectory(src/weight-server ${PROJECT_SOURCE_DIR}/build/weight-server)
else()
    message("Weight server: NO")
endif()
