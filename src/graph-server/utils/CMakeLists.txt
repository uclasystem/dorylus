cmake_minimum_required(VERSION 3.5)


# Options.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)


# Include ZeroMQ.
set(ZMQ_INC /usr/local/include)
set(ZMQ_LIB /usr/local/lib/libzmq.a)
include_directories(${ZMQ_INC})


# Include ZooKeeper
set(ZK_INC /usr/local/include)
set(ZK_LIB /usr/local/lib/libzookeeper_mt.a)
include_directories(${ZK_INC})


# Include Threads lib.
find_package(Threads REQUIRED)


# Include Boost lib.
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})


# Add the library objects.
add_library(utils "utils.cpp")
target_link_libraries(utils PUBLIC ${ZMQ_LIB} ${ZK_LIB} Threads::Threads ${BOOST_LIBRARIES})
target_compile_options(utils PRIVATE "-Wall" "-Werror" "-MMD")
