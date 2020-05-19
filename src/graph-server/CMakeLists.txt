cmake_minimum_required(VERSION 3.5)

FIND_PACKAGE(OpenMP)
# Options.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DVERBOSE_ERRORS -fno-omit-frame-pointer -fsanitize=address")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Include AWSSDK
if (BACKEND STREQUAL lambda)
    find_package(AWSSDK REQUIRED COMPONENTS lambda)
endif()


# Subdirectories, each needs a CMakeLists.txt.
if(BACKEND STREQUAL gpu)
    add_definitions(-D_GPU_ENABLED_)
    add_subdirectory(GPU-Computation)
endif()
if(BACKEND STREQUAL cpu)
    add_definitions(-D_CPU_ENABLED_)
endif()
if(BACKEND STREQUAL lambda)
    add_definitions(-D_LAMBDA_ENABLED_)
endif()
add_subdirectory(commmanager)
add_subdirectory(nodemanager)
add_subdirectory(engine)
add_subdirectory(parallel)
add_subdirectory(utils)
add_subdirectory(graph)

# Add the executable.
aux_source_directory(. GRAPH_SRCS)
add_executable(graphserver ${GRAPH_SRCS})

target_link_libraries(graphserver PRIVATE commmanager graph engine nodemanager threadpool utils common
                                  PUBLIC ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES}
                                        ${OBLIB} ${CBLIB})
target_compile_options(graphserver PRIVATE "-Wall" "-Werror" "-Wno-reorder" "-MMD")
