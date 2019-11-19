if(BUILD_GPU)
	enable_language(CUDA)
	find_library(CUDNN NAME libcudnn.so HINTS /opt/cuda/targets/x86_64-linux/lib/ /usr/local/cuda-10.0/lib64/)

	FIND_PACKAGE(CUDA REQUIRED)
	include_directories(${CUDA_INCLUDE_DIRS})
	cuda_add_library(CuMatrix SHARED "cu_matrix.cu")
	set_property(TARGET CuMatrix PROPERTY POSITION_INDEPENDENT_CODE ON)
	target_link_libraries(CuMatrix ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDNN} matrix)

	cuda_add_library(ComputingUnit SHARED "comp_unit.cu")
	set_property(TARGET ComputingUnit PROPERTY POSITION_INDEPENDENT_CODE ON)
	target_link_libraries(ComputingUnit CuMatrix)

	add_library(MessageService "message_service.cpp")
	set_property(TARGET MessageService PROPERTY POSITION_INDEPENDENT_CODE ON)
	target_link_libraries(MessageService ${ZMQ_LIB} matrix utils Threads::Threads)	

	cuda_add_library(ComputingServer SHARED "comp_server.cu")
	set_property(TARGET ComputingServer PROPERTY POSITION_INDEPENDENT_CODE ON)
	target_link_libraries(ComputingServer
						MessageService
						${Boost_LIBRARIES}
						ComputingUnit CuMatrix utils)
endif()
