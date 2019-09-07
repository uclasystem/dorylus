cmake_minimum_required(VERSION 3.5)
enable_language(CUDA)

#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DVERBOSE_ERRORS -fno-omit-frame-pointer -fsanitize=address")
set(CUDA_64_BIT_DEVICE_CODE ON)


# Project name.
project(GPU-Computation LANGUAGES CXX CUDA)

FIND_PACKAGE(CUDA REQUIRED)

# Include Boost lib.
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# Include ZeroMQ.
set(ZMQ_INC /usr/include)
set(ZMQ_LIB /usr/lib/libzmq.so)
include_directories(${ZMQ_INC})

# Include Threads.
find_package(Threads REQUIRED)

# Include BLAS.
set(OBLIB /usr/local/blas/OpenBLAS/lib/libopenblas.a)
set(CBLIB /usr/local/blas/CBLAS/lib/cblas_LINUX.a)
set(CBINC /usr/local/blas/CBLAS/include)
include_directories(${CBINC})

add_subdirectory(components)

cuda_add_executable(GPU-Computation GPU-Computation.cu)
target_link_libraries(GPU-Computation ComputingServer)
