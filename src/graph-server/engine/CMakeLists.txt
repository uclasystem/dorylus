cmake_minimum_required(VERSION 3.5)

aux_source_directory(ops OPS_SRC)
add_library(engine "engine.cpp" "utils.cpp" ${OPS_SRC})

if(BACKEND STREQUAL gpu)
    enable_language(CUDA)
    find_library(CUDNN NAME libcudnn.so HINTS /opt/cuda/targets/x86_64-linux/lib/ /usr/local/cuda-10.0/lib64/)
    FIND_PACKAGE(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries(engine PRIVATE ComputingUnit graph commmanager weight_comm gpu_comm nodemanager threadpool utils
                            PUBLIC  ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES} ${OpenMP_CXX_FLAGS})
endif()
if(BACKEND STREQUAL lambda)
    target_link_libraries(engine PRIVATE graph commmanager weight_comm lambda_comm nodemanager threadpool utils
                            PUBLIC  ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES} ${OpenMP_CXX_FLAGS})
endif()
if(BACKEND STREQUAL cpu)
    target_link_libraries(engine PRIVATE graph commmanager weight_comm cpu_comm nodemanager threadpool utils
                            PUBLIC  ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES} ${OpenMP_CXX_FLAGS})
endif()
target_compile_options(engine PRIVATE "-Wall" "-Werror" "-Wno-sign-compare" "-Wno-reorder" "-MMD" ${OpenMP_CXX_FLAGS})
