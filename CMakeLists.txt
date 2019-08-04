cmake_minimum_required(VERSION 3.5)


# Project name.
project(gnn-lambda LANGUAGES CXX)


# # Graph server (Data server).
# add_subdirectory(src/graph-server ${PROJECT_SOURCE_DIR}/build/graph-server)


# Coordination server.
add_subdirectory(src/coord-server ${PROJECT_SOURCE_DIR}/build/coord-server)


# Weight server.
add_subdirectory(src/weight-server ${PROJECT_SOURCE_DIR}/build/weight-server)


# Move executables into build.
# file(RENAME ${PROJECT_SOURCE_DIR}/build/graph-server/graphserver ${PROJECT_SOURCE_DIR}/build/graphserver)
file(RENAME ${PROJECT_SOURCE_DIR}/build/coord-server/coordserver ${PROJECT_SOURCE_DIR}/build/coordserver)
file(RENAME ${PROJECT_SOURCE_DIR}/build/weight-server/weightserver ${PROJECT_SOURCE_DIR}/build/weightserver)
