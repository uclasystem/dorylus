set(CUBLAS_LIB /opt/cuda/targets/x86_64-linux/lib/libcublas.so)

cuda_add_library(CuMatrix "cu_matrix.cu")
target_link_libraries(CuMatrix ${CUBLAS_LIB})

cuda_add_library(ComputingUnit "comp_unit.cu")
target_link_libraries(ComputingUnit CuMatrix ${CUBLAS_LIB})


cuda_add_library(utils ../../../src/graph-server/utils/utils.cpp)

cuda_add_library(ComputingServer "comp_server.cu")
target_link_libraries(ComputingServer 
					${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES}
					ComputingUnit CuMatrix utils)

