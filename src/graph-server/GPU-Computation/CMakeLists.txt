enable_language(CUDA)
set(CUDA_64_BIT_DEVICE_CODE ON)
FIND_PACKAGE(CUDA REQUIRED)

set(CUBLAS_LIB /opt/cuda/targets/x86_64-linux/lib/libcublas.so)

cuda_add_library(CuMatrix SHARED "cu_matrix.cu")
target_link_libraries(CuMatrix ${CUBLAS_LIB})

cuda_add_library(ComputingUnit SHARED "comp_unit.cu")
target_link_libraries(ComputingUnit CuMatrix ${CUBLAS_LIB})

cuda_add_library(ComputingServer SHARED "comp_server.cu")
target_link_libraries(ComputingServer 
					${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES}
					ComputingUnit CuMatrix utils)

