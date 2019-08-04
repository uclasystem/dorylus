cmake_minimum_required(VERSION 3.5)


# Project name.
project(gnn-lambda LANGUAGES CXX)


# # Graph server (Data server).
# add_subdirectory(src/graph-server ${PROJECT_SOURCE_DIR}/build/graph-server)


# Coordination server.
add_subdirectory(src/coord-server ${PROJECT_SOURCE_DIR}/build/coord-server)


# Weight server.
add_subdirectory(src/weight-server ${PROJECT_SOURCE_DIR}/build/weight-server)
