cmake_minimum_required(VERSION 3.5)


# Options.
set(CMAKE_CXX_STANDARD 11)


# Project name.
project(dnnl-forward-gcn LANGUAGES CXX)


# Include ZeroMQ.
set(ZMQ_INC /usr/local/include ../../src/common)
set(ZMQ_LIB /usr/local/lib/libzmq.a)
include_directories(${ZMQ_INC})

# Include BLAS.
set(OBLIB /usr/local/blas/OpenBLAS/lib/libopenblas.a)
set(CBLIB /usr/local/blas/CBLAS/lib/cblas_LINUX.a)
set(CBINC /usr/local/blas/CBLAS/include)
include_directories(${CBINC})

# Include Threads.
find_package(Threads REQUIRED)

find_library(MKL_LIB libmkldnn.so HINTS /usr/local/dnnl/lib)
set(MKL_INC /usr/local/dnnl/include)
include_directories(${MKL_INC})

# Include AWS Dependencies
find_package(aws-lambda-runtime REQUIRED)
find_package(AWSSDK REQUIRED core)


add_subdirectory(../../common common)

# Create the executable target.
aux_source_directory(. FUNC_SRC)
add_executable(${PROJECT_NAME} ${FUNC_SRC})
target_link_libraries(${PROJECT_NAME} PUBLIC common Threads::Threads AWS::aws-lambda-runtime ${MKL_LIB} ${ZMQ_LIB} ${OBLIB} ${CLIB})


# Create the packaging target.
aws_lambda_package_target(${PROJECT_NAME})
