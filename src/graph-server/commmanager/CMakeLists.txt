cmake_minimum_required(VERSION 3.5)

# Add the library objects.
add_library(commmanager "commmanager.cpp")
target_link_libraries(commmanager PRIVATE utils
                                  PUBLIC ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(commmanager PRIVATE "-Wall" "-Werror" "-Wno-unused-but-set-variable" "-MMD")

add_library(weight_comm "weight_comm.cpp")
target_link_libraries(weight_comm PRIVATE utils
                                  PUBLIC ${ZMQ_LIB} ${Boost_LIBRARIES})
target_compile_options(weight_comm PRIVATE "-Wall" "-Werror" "-Wno-unused-but-set-variable" "-MMD")

if (BACKEND STREQUAL lambda)
    add_library(lambda_comm SHARED "lambda_comm.cpp" "resource_comm.cpp" "lambdaworker.cpp")
    set_property(TARGET lambda_comm PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(lambda_comm PRIVATE utils common
                                      PUBLIC  ${AWSSDK_LINK_LIBRARIES} ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
    target_compile_options(lambda_comm PRIVATE "-Wall" "-Werror" "-Wno-reorder" "-MMD")
endif()


if (BACKEND STREQUAL cpu)
    add_library(MessageService "message_service.cpp")
    set_property(TARGET MessageService PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(MessageService ${ZMQ_LIB} common utils Threads::Threads)

    find_package(OpenMP REQUIRED)
    add_library(cpu_comm SHARED "CPU_comm.cpp" "resource_comm.cpp")
    set_property(TARGET cpu_comm PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(cpu_comm PRIVATE utils common MessageService ${OpenMP_CXX_FLAGS}
                                    PUBLIC  ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES} )
    target_compile_options(cpu_comm PRIVATE "-Wall" "-Werror" "-Wno-reorder" "-MMD" ${OpenMP_CXX_FLAGS})
endif()


if(BACKEND STREQUAL gpu)
    add_library(MessageService "message_service.cpp")
    set_property(TARGET MessageService PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(MessageService ${ZMQ_LIB} common utils Threads::Threads)

    enable_language(CUDA)
    FIND_PACKAGE(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_library(gpu_comm SHARED "GPU_comm.cpp" "resource_comm.cpp")
    set_property(TARGET gpu_comm PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(gpu_comm PRIVATE utils MessageService ComputingServer
                                  PUBLIC  ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
    target_compile_options(gpu_comm PRIVATE "-Wall" "-Werror" "-Wno-reorder" "-MMD")
endif()
