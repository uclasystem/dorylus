cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 11)
project(weightserver LANGUAGES CXX)


execute_process(COMMAND whoami OUTPUT_VARIABLE USER)
string(REGEX REPLACE "\n$" "" USER "${USER}")	# Remove new-line char

set(INST /home/${USER}/gnn-lambda/installs/out)

set(ZMQ_INC ${INST}/include)
set(ZMQ_LIB ${INST}/lib/libzmq.a)

include_directories( ${ZMQ_INC} )
find_package(Threads REQUIRED)
add_executable(${PROJECT_NAME} "weightserver.cpp")
target_link_libraries(${PROJECT_NAME} PUBLIC ${ZMQ_LIB} Threads::Threads)

