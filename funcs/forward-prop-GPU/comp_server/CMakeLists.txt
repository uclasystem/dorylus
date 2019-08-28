cuda_add_library (ComputingServer "comp_server.cu")
target_link_libraries(ComputingServer ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES})