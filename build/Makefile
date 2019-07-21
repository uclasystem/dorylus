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
SRC_PATH        := ../src
NODE_PATH       := $(SRC_PATH)/nodemanager
COMM_PATH       := $(SRC_PATH)/commmanager
ENGINE_PATH     := $(SRC_PATH)/engine
PARALLEL_PATH   := $(SRC_PATH)/parallel
UTILS_PATH      := $(SRC_PATH)/utils
BM_PATH         := $(SRC_PATH)/benchmarks

# Build paths
BUILD_PATH      := .
OBJ_PATH        := $(BUILD_PATH)/objs

# Benchmarks
AGG_TARGET      := aggregate
AGG_TARGET_BIN  := $(AGG_TARGET).bin

# Libraries source
OMP_LIBS        := -fopenmp
BOOST_LIBS      := -lboost_system -lboost_filesystem -lboost_program_options
ZK_LIBS         := -lzookeeper_mt
ZK_LIBPATH      := /home/$(USER)/aspire-streaming/installs/out/lib
ZMQ_LIBS        := -lzmq
ZMQ_LIBPATH     := /home/$(USER)/aspire-streaming/installs/out/lib
LIBS            := -L$(ZK_LIBPATH) $(ZK_LIBS) -L$(ZMQ_LIBPATH) $(ZMQ_LIBS) -lpthread $(OMP_LIBS) $(BOOST_LIBS)

# Libraries linking
ZK_INCPATH      := /usr/local/include/zookeeper
ZMQ_INCPATH     := /home/$(USER)/aspire-streaming/installs/out/include
INC             := -I$(ZK_INCPATH) -I$(ZMQ_INCPATH)
LDFLAGS         := -Wl,-rpath=$(ZK_LIBPATH) -Wl,-rpath=$(ZMQ_LIBPATH)

# Objects
BASE_OBJS       := $(OBJ_PATH)/commmanager.o $(OBJ_PATH)/nodemanager.o $(OBJ_PATH)/zkinterface.o $(OBJ_PATH)/zkhelper.o $(OBJ_PATH)/utils.o $(OBJ_PATH)/threadpool.o
AGG_OBJS        := $(BASE_OBJS) $(OBJ_PATH)/$(AGG_TARGET).o


#
# MAKE targets
#

# Compiler & Flags
CPP     := g++ -std=c++11
CFLAGS  := -DTHREADED -Wall -Wno-sign-compare -MMD
ifeq ($(mode),release)
	CFLAGS += -O3
else
	CFLAGS += -DVERBOSE_ERRORS -g -fsanitize=address
endif

# Overall batch target
all: prompt directories agg_tgt

# Benchmark individual targets
$(AGG_TARGET):      prompt directories agg_tgt

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

# Detailed target rules
agg_tgt: $(AGG_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFALGS) -o $(AGG_TARGET_BIN)

$(OBJ_PATH)/%.o: $(NODE_PATH)/%.cpp $(NODE_PATH)/%.h
	$(CPP) $(CFLAGS) $(INC) $(LIBS) $(LDFLAGS) $(INC) -c $< -o $@

$(OBJ_PATH)/%.o: $(COMM_PATH)/%.cpp $(COMM_PATH)/%.h
	$(CPP) $(CFLAGS) $(INC) $(LIBS) $(LDFLAGS) $(INC) -c $< -o $@

$(OBJ_PATH)/%.o: $(ENGINE_PATH)/%.cpp $(ENGINE_PATH)/%.h
	$(CPP) $(CFLAGS) $(INC) $(LIBS) $(LDFLAGS) $(INC) -c $< -o $@

$(OBJ_PATH)/%.o: $(PARALLEL_PATH)/%.cpp $(PARALLEL_PATH)/%.h
	$(CPP) $(CFLAGS) $(INC) $(LIBS) $(LDFLAGS) $(INC) -c $< -o $@

$(OBJ_PATH)/%.o: $(UTILS_PATH)/%.cpp $(UTILS_PATH)/%.h
	$(CPP) $(CFLAGS) $(INC) $(LIBS) $(LDFLAGS) $(INC) -c $< -o $@

$(OBJ_PATH)/%.o: $(BM_PATH)/%.cpp $(BM_PATH)/%.h
	$(CPP) $(CFLAGS) $(INC) $(LIBS) $(LDFLAGS) $(INC) -c $< -o $@

$(OBJ_PATH)/%.o: $(BM_PATH)/%.cpp
	$(CPP) $(CFLAGS) $(INC) $(LIBS) $(LDFLAGS) $(INC) -c $< -o $@

# Clean
.PHONY: clean
clean:
	rm -rf $(OBJ_PATH) $(AGG_TARGET_BIN)

