cmake_minimum_required(VERSION 3.5)


# Options.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DVERBOSE_ERRORS -fno-omit-frame-pointer -fsanitize=address -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Project name.
project(backward LANGUAGES CXX)

# Include ZeroMQ.
set(ZMQ_INC /usr/local/include ../../src/common)
set(ZMQ_LIB /usr/local/lib/libzmq.a)
include_directories(${ZMQ_INC})

# Include BLAS.
set(OBLIB /usr/local/blas/OpenBLAS/lib/libopenblas.a)
set(CBLIB /usr/local/blas/CBLAS/lib/cblas_LINUX.a)
set(CBINC /usr/local/blas/CBLAS/include)
include_directories(${CBINC})

# Include AWS SDK for Json Utils
find_package(AWSSDK REQUIRED COMPONENTS core)

# Include Threads.
find_package(Threads REQUIRED)

# Include AWS Lambda runtime.
find_package(aws-lambda-runtime REQUIRED)

add_subdirectory(../../common common)

# Create the executable target.
aux_source_directory(. FUNC_SRC)
add_executable(${PROJECT_NAME} ${FUNC_SRC})
target_link_libraries(${PROJECT_NAME} PUBLIC common Threads::Threads ${AWSSDK_LINK_LIBRARIES} AWS::aws-lambda-runtime ${OBLIB} ${CBLIB} ${ZMQ_LIB})


# Create the packaging target.
aws_lambda_package_target(${PROJECT_NAME})
