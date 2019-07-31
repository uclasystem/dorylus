cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)
project(coordination_server LANGUAGES CXX)


execute_process(COMMAND whoami OUTPUT_VARIABLE USER)
string(REGEX REPLACE "\n$" "" USER "${USER}")	# Remove new-line char


set(INST /home/${USER}/gnn-lambda/installs/out)
set(ZMQ_INC ${INST}/include)
set(ZMQ_LIB ${INST}/lib/libzmq.a)


include_directories( ${ZMQ_INC} )

find_package(Threads REQUIRED)
find_package(AWSSDK REQUIRED COMPONENTS lambda)


add_executable(${PROJECT_NAME} "coordserver.cpp")
target_link_libraries(${PROJECT_NAME} PUBLIC ${ZMQ_LIB} ${AWSSDK_LINK_LIBRARIES} Threads::Threads)
target_compile_options(${PROJECT_NAME} PRIVATE "-Wall" "-Werror")
