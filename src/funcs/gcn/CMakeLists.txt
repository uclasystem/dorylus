cmake_minimum_required(VERSION 3.5)


# Options.
set(CMAKE_CXX_STANDARD 11)


# Project name.
project(gcn LANGUAGES CXX)


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

# Include AWS Lambda runtime.
find_package(aws-lambda-runtime REQUIRED)

# Include AWS SDK for Json Utils
find_package(AWSSDK REQUIRED COMPONENTS core)


add_subdirectory(../../common common)
add_subdirectory(ops)

# Create the executable target.

add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE ops PUBLIC common Threads::Threads ${AWSSDK_LINK_LIBRARIES} AWS::aws-lambda-runtime ${OBLIB} ${CBLIB} ${ZMQ_LIB})


# Create the packaging target.
aws_lambda_package_target(${PROJECT_NAME})
