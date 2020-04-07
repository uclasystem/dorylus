cmake_minimum_required(VERSION 3.5)


# Add the library objects.
add_library(edge "edge.cpp")
target_link_libraries(edge PRIVATE utils
                           PUBLIC ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(edge PRIVATE "-Wall" "-Werror" "-Wno-sign-compare" "-Wno-reorder" "-MMD")

add_library(vertex "vertex.cpp")
target_link_libraries(vertex PRIVATE utils
                             PUBLIC ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(vertex PRIVATE "-Wall" "-Werror" "-Wno-sign-compare" "-Wno-reorder" "-MMD")

add_library(graph "graph.cpp")
target_link_libraries(graph PRIVATE edge vertex utils
                            PUBLIC ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})

target_compile_options(graph PRIVATE "-Wall" "-Werror" "-Wno-sign-compare" "-Wno-reorder" "-MMD")

add_library(dataloader "dataloader.cpp")
target_link_libraries(dataloader PRIVATE edge vertex utils
                                 PUBLIC ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
target_compile_options(dataloader PRIVATE "-Wall" "-Werror" "-Wno-sign-compare" "-Wno-reorder" "-MMD")

add_library(engine "engine.cpp")
if(BUILD_GPU)
    enable_language(CUDA)
    find_library(CUDNN NAME libcudnn.so HINTS /opt/cuda/targets/x86_64-linux/lib/ /usr/local/cuda-10.0/lib64/)
    FIND_PACKAGE(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries(engine PRIVATE ComputingUnit edge vertex graph dataloader commmanager weight_comm GPU_comm nodemanager threadpool utils
                             PUBLIC  ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})
else()
target_link_libraries(engine PRIVATE edge vertex graph dataloader commmanager weight_comm lambda_comm nodemanager threadpool utils
                             PUBLIC  ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES} ${OpenMP_CXX_FLAGS})
endif()
target_compile_options(engine PRIVATE "-Wall" "-Werror" "-Wno-sign-compare" "-Wno-reorder" "-MMD" ${OpenMP_CXX_FLAGS})
