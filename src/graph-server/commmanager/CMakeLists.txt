cmake_minimum_required(VERSION 3.5)

# Add the library objects.
add_library(commmanager "commmanager.cpp")
target_link_libraries(commmanager PRIVATE utils
                                  PUBLIC ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(commmanager PRIVATE "-Wall" "-Werror" "-Wno-unused-but-set-variable" "-MMD")

add_library(resource_comm "resource_comm.cpp")
target_link_libraries(resource_comm PRIVATE utils libdl.so)
target_compile_options(resource_comm PRIVATE "-Wall" "-Werror" "-Wno-unused-but-set-variable" "-MMD")


add_library(lambda_comm SHARED "lambda_comm.cpp" "lambdaworker.cpp")
set_property(TARGET lambda_comm PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(lambda_comm PRIVATE utils resource_comm
                                  PUBLIC  ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(lambda_comm PRIVATE "-Wall" "-Werror" "-Wno-reorder" "-MMD")


add_library(gpu_comm SHARED "GPU_comm.cpp" )
set_property(TARGET gpu_comm PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(gpu_comm PRIVATE utils resource_comm ComputingServer
                              PUBLIC  ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(gpu_comm PRIVATE "-Wall" "-Werror" "-Wno-reorder" "-MMD")