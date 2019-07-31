cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 11)
project(forward-prop-josehu LANGUAGES CXX)


execute_process(COMMAND whoami OUTPUT_VARIABLE USER)
string(REGEX REPLACE "\n$" "" USER "${USER}")	# Remove new-line char

set(INST /usr/local)

set(ZMQ_INC ${INST}/include)
set(ZMQ_LIB ${INST}/lib/libzmq.a)

# BLAS lib/inc deps
set(BLLIB /usr/lib64/libopenblas.so.0)
set(CBLIB ${INST}/blas/CBLAS/lib/cblas_LINUX.a)
set(CBINC ${INST}/blas/CBLAS/include)

# Boost lib/inc deps
set(Boost_INCLUDE_DIRS /usr/local/include/boost)
set(Boost_LIBRARY_DIRS /usr/local/lib)

include_directories( ${Boost_INCLUDE_DIRS} ${CBINC} )
find_package(Boost REQUIRED)
find_package(Threads REQUIRED)
find_package(aws-lambda-runtime REQUIRED)
add_executable(${PROJECT_NAME} "forward_prop_josehu.cpp")
target_link_libraries(${PROJECT_NAME} PUBLIC ${Boost_LIBRARIES}
		Threads::Threads AWS::aws-lambda-runtime ${BLLIB}
		${CBLIB} ${ZMQ_LIB})


aws_lambda_package_target(${PROJECT_NAME})
