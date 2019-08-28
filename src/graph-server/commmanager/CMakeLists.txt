cmake_minimum_required(VERSION 3.5)


# Add the library objects.
add_library(commmanager "commmanager.cpp")
target_link_libraries(commmanager PRIVATE utils
                                  PUBLIC ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(commmanager PRIVATE "-Wall" "-Werror" "-Wno-unused-but-set-variable" "-MMD")
add_library(lambda_comm "lambda_comm.cpp" "lambdaworker.cpp")
target_link_libraries(lambda_comm PRIVATE utils
                                  PUBLIC  ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(lambda_comm PRIVATE "-Wall" "-Werror" "-Wno-reorder" "-MMD")

add_library(gpu_comm "GPU_comm.cpp" )
target_link_libraries(gpu_comm PRIVATE utils
                              PUBLIC  ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(gpu_comm PRIVATE "-Wall" "-Werror" "-Wno-reorder" "-MMD")