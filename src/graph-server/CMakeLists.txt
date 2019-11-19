cmake_minimum_required(VERSION 3.5)


# Options.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DVERBOSE_ERRORS -fno-omit-frame-pointer -fsanitize=address")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()


# Subdirectories, each needs a CMakeLists.txt.
if(BUILD_GPU)
	add_definitions(-D_GPU_ENABLED_)
    add_subdirectory(GPU-Computation)
endif()
add_subdirectory(commmanager)
add_subdirectory(nodemanager)
add_subdirectory(engine)
add_subdirectory(parallel)
add_subdirectory(utils)

# Add the executable.
aux_source_directory(. GRAPH_SRCS)
add_executable(graphserver ${GRAPH_SRCS})

target_link_libraries(graphserver PRIVATE commmanager resource_comm edge vertex graph engine nodemanager threadpool utils matrix
                                  PUBLIC ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES}
                                        ${OBLIB} ${CBLIB})
target_compile_options(graphserver PRIVATE "-Wall" "-Werror" "-Wno-reorder" "-MMD")
