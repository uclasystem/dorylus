cmake_minimum_required(VERSION 3.5)
enable_language(CUDA)

#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DVERBOSE_ERRORS -fno-omit-frame-pointer -fsanitize=address")
set(CUDA_64_BIT_DEVICE_CODE ON)

# Project name.
project(forward-GPU LANGUAGES CXX CUDA)

FIND_PACKAGE(CUDA REQUIRED)

# Include Boost lib.
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# Include ZeroMQ.
set(ZMQ_INC /usr/local/include)
set(ZMQ_LIB /usr/local/lib/libzmq.a)
include_directories(${ZMQ_INC})

# Include Threads.
find_package(Threads REQUIRED)

add_subdirectory(comp_unit)
add_subdirectory(comp_server)
cuda_add_executable(forward_GPU forward_GPU.cu)
#add_executable(forward_GPU forward_GPU.cpp)
target_link_libraries(forward_GPU ComputingServer ComputingUnit)
