enable_language(CXX)
add_library (ComputingServer "comp_server.cpp")
target_link_libraries(ComputingServer PUBLIC ${ZMQ_LIB} Threads::Threads )