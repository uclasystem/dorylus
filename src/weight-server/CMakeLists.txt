cmake_minimum_required(VERSION 3.5)


# Options.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DVERBOSE_ERRORS -fno-omit-frame-pointer -fsanitize=address")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add the executable.
aux_source_directory(. WEIGHT_SRCS)


add_executable(weightserver ${WEIGHT_SRCS})
target_link_libraries(weightserver PRIVATE common
                                   PUBLIC ${ZMQ_LIB} Threads::Threads ${Boost_LIBRARIES} ${OBLIB} ${CBLIB})
target_compile_options(weightserver PRIVATE "-Wall" "-Werror" "-Wno-reorder")
