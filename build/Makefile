#
# Makefile for building ASPIRE. Make sure everything has been setup according to
# README, strictly in the designed place.
#


#
# Alias
#

# Get your username
USER            := $(shell whoami)

# Source paths
SRC_PATH        := ../src/graph-server
NODE_PATH       := $(SRC_PATH)/nodemanager
COMM_PATH       := $(SRC_PATH)/commmanager
ENGINE_PATH     := $(SRC_PATH)/engine
PARALLEL_PATH   := $(SRC_PATH)/parallel
UTILS_PATH      := $(SRC_PATH)/utils

# Build paths
BUILD_PATH      := .
OBJ_PATH        := $(BUILD_PATH)/objs

# Libraries source
OMP_LIBS        := -fopenmp
BOOST_LIBS      := -lboost_system -lboost_filesystem -lboost_program_options
ZK_LIBS         := -lzookeeper_mt
ZK_LIBPATH      := /home/$(USER)/gnn-lambda/installs/out/lib
ZMQ_LIBS        := -lzmq
ZMQ_LIBPATH     := /home/$(USER)/gnn-lambda/installs/out/lib
LIBS            := -L$(ZK_LIBPATH) $(ZK_LIBS) -L$(ZMQ_LIBPATH) $(ZMQ_LIBS) -lpthread $(OMP_LIBS) $(BOOST_LIBS)

# Libraries headers & linking
ZK_INCPATH      := /usr/local/include/zookeeper
ZMQ_INCPATH     := /home/$(USER)/gnn-lambda/installs/out/include
INC             := -I$(ZK_INCPATH) -I$(ZMQ_INCPATH)
LDFLAGS         := -Wl,-rpath=$(ZK_LIBPATH) -Wl,-rpath=$(ZMQ_LIBPATH)


#
# MAKE targets
#

# Compiler & Flags
CXX         := g++
CXXFLAGS    := -Wall -Wno-sign-compare -Wno-unused-function -MMD -std=c++11
ifeq ($(mode),release)
	CXXFLAGS += -O3
else
	CXXFLAGS += -DVERBOSE_ERRORS -g -fno-omit-frame-pointer -fsanitize=address
	LIBS += -fsanitize=address
endif

# Targets for you to choose
all: prompt directories gnn-lambda.bin

# Phony utilities
prompt:
ifneq ($(mode),release)
ifneq ($(mode),debug)
	@echo "\e[31;1mERROR: Invalid build mode. Please specify either \"mode=release\" or \"mode=debug\".\e[0m"
	@exit 1
endif
endif
	@echo "\e[33;1mBuilding: On \"$(mode)\" mode.....\e[0m"

directories:
	mkdir -p $(OBJ_PATH) 

# Frameowrk objects internal targets
BASE_OBJS   := $(OBJ_PATH)/lambda_comm.o $(OBJ_PATH)/commmanager.o $(OBJ_PATH)/nodemanager.o $(OBJ_PATH)/engine.o $(OBJ_PATH)/graph.o $(OBJ_PATH)/vertex.o $(OBJ_PATH)/edge.o $(OBJ_PATH)/zkinterface.o $(OBJ_PATH)/utils.o $(OBJ_PATH)/threadpool.o

$(OBJ_PATH)/%.o: $(NODE_PATH)/%.cpp $(NODE_PATH)/%.hpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(OBJ_PATH)/%.o: $(COMM_PATH)/%.cpp $(COMM_PATH)/%.hpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(OBJ_PATH)/%.o: $(ENGINE_PATH)/%.cpp $(ENGINE_PATH)/%.hpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(OBJ_PATH)/%.o: $(PARALLEL_PATH)/%.cpp $(PARALLEL_PATH)/%.hpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(OBJ_PATH)/%.o: $(UTILS_PATH)/%.cpp $(UTILS_PATH)/%.hpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

# Main executable target
$(OBJ_PATH)/main.o: $(SRC_PATH)/main.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

gnn-lambda.bin: $(OBJ_PATH)/main.o $(BASE_OBJS)
	$(CXX) $^ $(LIBS) $(LDFLAGS) -o $@

# Clean
.PHONY: clean
clean:
	rm -rf $(OBJ_PATH) gnn-lambda.bin
