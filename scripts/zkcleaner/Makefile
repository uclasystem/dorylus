CZK_TARGET       := cleanzookeeper
CZK_TARGET_BIN   := $(CZK_TARGET).bin

SRC_PATH        := ../../src
NODE_PATH       := $(SRC_PATH)/nodemanager
UTILS_PATH      := $(SRC_PATH)/utils

BUILD_PATH      := .
OBJ_PATH        := $(BUILD_PATH)/objs

CPP             := g++ -std=c++11
RM              := rm -rf
MK              := mkdir -p

# Flags
CFLAGS          := -DTHREADED -Wall -MMD -O3 ${EFLAGS}

ZK_LIBS         := -lzookeeper_mt
ZK_LIBPATH      := /home/keval/Desktop/workspace/installs/builds/zookeeper-cbinding/lib

LIBS            := -L$(ZK_LIBPATH) $(ZK_LIBS)  -lpthread -lboost_program_options

ZK_INCPATH      := /home/keval/Desktop/workspace/installs/builds/zookeeper-cbinding/include/zookeeper
ZMQ_INCPATH     := /home/keval/Desktop/workspace/installs/builds/zeromq-4.1.2/include
INC             := -I$(ZK_INCPATH) -I$(ZMQ_INCPATH)

LDFLAGS         := -Wl,-rpath=$(ZK_LIBPATH)

BASE_OBJS       := $(OBJ_PATH)/zkinterface.o $(OBJ_PATH)/zkhelper.o $(OBJ_PATH)/utils.o 

CZK_OBJS         := $(BASE_OBJS) $(OBJ_PATH)/cleanzookeeper.o

all: directories czk_tgt

#$(CZK_TARGET): directories czk_tgt

directories:
	$(MK) $(OBJ_PATH) 

czk_tgt: $(CZK_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(CZK_TARGET_BIN)

$(OBJ_PATH)/%.o: $(NODE_PATH)/%.cpp $(NODE_PATH)/%.h
	$(CPP) $(CFLAGS) $(INC) $(LIBS) $(LDFLAGS) $(INC) -c $< -o $@

$(OBJ_PATH)/%.o: $(UTILS_PATH)/%.cpp $(UTILS_PATH)/%.h
	$(CPP) $(CFLAGS) $(INC) $(LIBS) $(LDFLAGS) $(INC) -c $< -o $@

$(OBJ_PATH)/%.o: %.cpp
	$(CPP) $(CFLAGS) $(INC) $(LIBS) $(LDFLAGS) $(INC) -c $< -o $@

clean:
	$(RM) $(OBJ_PATH) $(CZK_TARGET_BIN)

