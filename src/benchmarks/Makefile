PR_TARGET       := pagerank
SSSP_TARGET     := sssp
ASSSP_TARGET    := asssp
TSSSP_TARGET    := tsssp
BFS_TARGET      := bfs
ABFS_TARGET     := abfs
TBFS_TARGET     := tbfs
MSSP_TARGET	    := mssp
SSWP_TARGET     := sswp
ASSWP_TARGET    := asswp
TSSWP_TARGET    := tsswp
KCORE_TARGET	  := kcore
CC_TARGET	      := connectedcomponents
ACC_TARGET      := aconnectedcomponents
TCC_TARGET      := tconnectedcomponents
CD_TARGET       := communitydetection
ACD_TARGET      := acommunitydetection
ABCD_TARGET     := abcommunitydetection
CDP2_TARGET     := communitydetectionp2
DEG_TARGET      := degree

PR_TARGET_BIN     := $(PR_TARGET).bin
SSSP_TARGET_BIN   := $(SSSP_TARGET).bin
ASSSP_TARGET_BIN  := $(ASSSP_TARGET).bin
TSSSP_TARGET_BIN  := $(TSSSP_TARGET).bin
BFS_TARGET_BIN    := $(BFS_TARGET).bin
ABFS_TARGET_BIN   := $(ABFS_TARGET).bin
TBFS_TARGET_BIN   := $(TBFS_TARGET).bin
MSSP_TARGET_BIN	  := $(MSSP_TARGET).bin
SSWP_TARGET_BIN   := $(SSWP_TARGET).bin
ASSWP_TARGET_BIN  := $(ASSWP_TARGET).bin
TSSWP_TARGET_BIN  := $(TSSWP_TARGET).bin
KCORE_TARGET_BIN	:= $(KCORE_TARGET).bin
CC_TARGET_BIN	    := $(CC_TARGET).bin
TCC_TARGET_BIN    := $(TCC_TARGET).bin
ACC_TARGET_BIN    := $(ACC_TARGET).bin
CD_TARGET_BIN     := $(CD_TARGET).bin
ACD_TARGET_BIN    := $(ACD_TARGET).bin
ABCD_TARGET_BIN   := $(ABCD_TARGET).bin
CDP2_TARGET_BIN   := $(CDP2_TARGET).bin
DEG_TARGET_BIN    := $(DEG_TARGET).bin

SRC_PATH        := ../src
NODE_PATH       := $(SRC_PATH)/nodemanager
COMM_PATH       := $(SRC_PATH)/commmanager
ENGINE_PATH     := $(SRC_PATH)/engine
PARALLEL_PATH   := $(SRC_PATH)/parallel
UTILS_PATH      := $(SRC_PATH)/utils
BM_PATH         := $(SRC_PATH)/benchmarks

BUILD_PATH      := .
OBJ_PATH        := $(BUILD_PATH)/objs

CPP             := g++ -std=c++11
RM              := rm -rf
MK              := mkdir -p

# Flags
EFLAGS          := -DVERBOSE_ERRORS #-DSTATICIP
CFLAGS          := -DTHREADED -Wall -MMD -O3 ${EFLAGS}

#OMP_LIBS       := -fopenmp
#BOOST_LIBS     := -lboost_system -lboost_filesystem
ZK_LIBS         := -lzookeeper_mt
ZK_LIBPATH      := /home/ubuntu/Desktop/workspace/installs/builds/zookeeper-cbinding/lib

ZMQ_LIBS        := -lzmq
ZMQ_LIBPATH     := /home/ubuntu/Desktop/workspace/installs/builds/zeromq-4.1.4/lib

LIBS            := -L$(ZK_LIBPATH) $(ZK_LIBS) -L$(ZMQ_LIBPATH) $(ZMQ_LIBS) -lpthread -fopenmp -lboost_program_options

ZK_INCPATH      := /home/ubuntu/Desktop/workspace/installs/builds/zookeeper-cbinding/include/zookeeper
ZMQ_INCPATH     := /home/ubuntu/Desktop/workspace/installs/builds/zeromq-4.1.4/include
INC             := -I$(ZK_INCPATH) -I$(ZMQ_INCPATH)

LDFLAGS         := -Wl,-rpath=$(ZK_LIBPATH) -Wl,-rpath=$(ZMQ_LIBPATH)

#BASE_OBJS       := $(COMM_PATH)/commmanager.o $(NODE_PATH)/nodemanager.o $(NODE_PATH)/zkinterface.o $(NODE_PATH)/zkhelper.o $(UTILS_PATH)/utils.o $(PARALLEL_PATH)/threadpool.o 

BASE_OBJS       := $(OBJ_PATH)/commmanager.o $(OBJ_PATH)/nodemanager.o $(OBJ_PATH)/zkinterface.o $(OBJ_PATH)/zkhelper.o $(OBJ_PATH)/utils.o $(OBJ_PATH)/threadpool.o

PR_OBJS         := $(BASE_OBJS) $(OBJ_PATH)/pagerank.o
SSSP_OBJS       := $(BASE_OBJS) $(OBJ_PATH)/sssp.o
ASSSP_OBJS      := $(BASE_OBJS) $(OBJ_PATH)/asssp.o
TSSSP_OBJS      := $(BASE_OBJS) $(OBJ_PATH)/tsssp.o
BFS_OBJS        := $(BASE_OBJS) $(OBJ_PATH)/bfs.o
ABFS_OBJS       := $(BASE_OBJS) $(OBJ_PATH)/abfs.o
TBFS_OBJS       := $(BASE_OBJS) $(OBJ_PATH)/tbfs.o
MSSP_OBJS	      := $(BASE_OBJS) $(OBJ_PATH)/mssp.o
SSWP_OBJS       := $(BASE_OBJS) $(OBJ_PATH)/sswp.o
ASSWP_OBJS      := $(BASE_OBJS) $(OBJ_PATH)/asswp.o
TSSWP_OBJS      := $(BASE_OBJS) $(OBJ_PATH)/tsswp.o
KCORE_OBJS	    := $(BASE_OBJS) $(OBJ_PATH)/kcore.o
CC_OBJS		      := $(BASE_OBJS) $(OBJ_PATH)/connectedcomponents.o
TCC_OBJS        := $(BASE_OBJS) $(OBJ_PATH)/tconnectedcomponents.o
ACC_OBJS        := $(BASE_OBJS) $(OBJ_PATH)/aconnectedcomponents.o
CD_OBJS         := $(BASE_OBJS) $(OBJ_PATH)/communitydetection.o
ACD_OBJS        := $(BASE_OBJS) $(OBJ_PATH)/acommunitydetection.o
ABCD_OBJS       := $(BASE_OBJS) $(OBJ_PATH)/abcommunitydetection.o
CDP2_OBJS       := $(BASE_OBJS) $(OBJ_PATH)/communitydetectionp2.o
DEG_OBJS        := $(BASE_OBJS) $(OBJ_PATH)/degree.o

#-include    $(PR_OBJS:%.o=%.d)

#all:        CFLAGS += -O3
all: directories pr_tgt sssp_tgt asssp_tgt tsssp_tgt bfs_tgt abfs_tgt tbfs_tgt mssp_tgt sswp_tgt asswp_tgt tsswp_tgt kcore_tgt cc_tgt tcc_tgt cd_tgt acd_tgt abcd_tgt cdp2_tgt deg_tgt 

$(PR_TARGET): directories pr_tgt
$(SSSP_TARGET): directories sssp_tgt
$(ASSSP_TARGET): directories asssp_tgt
$(TSSSP_TARGET): directories tsssp_tgt
$(BFS_TARGET): directories bfs_tgt
$(ABFS_TARGET): directories abfs_tgt
$(TBFS_TARGET): directories tbfs_tgt
$(MSSP_TARGET): directories mssp_tgt
$(SSWP_TARGET): directories sswp_tgt
$(ASSWP_TARGET): directories asswp_tgt
$(TSSWP_TARGET): directories tsswp_tgt
$(KCORE_TARGET): directories kcore_tgt
$(CC_TARGET): directories cc_tgt
$(TCC_TARGET): directories tcc_tgt
$(ACC_TARGET): directories acc_tgt
$(CD_TARGET): directories cd_tgt
$(ACD_TARGET): directories acd_tgt
$(ABCD_TARGET): directories abcd_tgt
$(CDP2_TARGET): directories cdp2_tgt
$(DEG_TARGET): directories deg_tgt

directories:
	$(MK) $(OBJ_PATH) 

pr_tgt: $(PR_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(PR_TARGET_BIN)

sssp_tgt: $(SSSP_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(SSSP_TARGET_BIN)

asssp_tgt: $(ASSSP_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(ASSSP_TARGET_BIN)

tsssp_tgt: $(TSSSP_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(TSSSP_TARGET_BIN)

bfs_tgt: $(BFS_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(BFS_TARGET_BIN)

abfs_tgt: $(ABFS_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(ABFS_TARGET_BIN)

tbfs_tgt: $(TBFS_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(TBFS_TARGET_BIN)

mssp_tgt: $(MSSP_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(MSSP_TARGET_BIN)

sswp_tgt: $(SSWP_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(SSWP_TARGET_BIN)

asswp_tgt: $(ASSWP_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(ASSWP_TARGET_BIN)

tsswp_tgt: $(TSSWP_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(TSSWP_TARGET_BIN)

kcore_tgt: $(KCORE_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(KCORE_TARGET_BIN)

cc_tgt: $(CC_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(CC_TARGET_BIN)

acc_tgt: $(ACC_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(ACC_TARGET_BIN)

tcc_tgt: $(TCC_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(TCC_TARGET_BIN)

cd_tgt: $(CD_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(CD_TARGET_BIN)

acd_tgt: $(ACD_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(ACD_TARGET_BIN)

abcd_tgt: $(ABCD_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(ABCD_TARGET_BIN)

cdp2_tgt: $(CDP2_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(CDP2_TARGET_BIN)

deg_tgt: $(DEG_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(DEG_TARGET_BIN)

#$(OBJ_PATH)/%.o: $(NODE_PATH)/%.cpp $(NODE_PATH)/%.h $(COMM_PATH)/%.cpp $(COMM_PATH)/%.h $(ENGINE_PATH)/%.cpp $(ENGINE_PATH)/%.h $(PARALLEL_PATH)/%.cpp $(PARALLEL_PATH)/%.h $(UTILS_PATH)/%.cpp $(UTILS_PATH)/%.h $(BM_PATH)/%.cpp $(BM_PATH)/%.h

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

clean:
	$(RM) $(OBJ_PATH) $(PR_TARGET_BIN) $(SSSP_TARGET_BIN) $(ASSSP_TARGET_BIN) $(TSSSP_TARGET_BIN) $(BFS_TARGET_BIN) $(ABFS_TARGET_BIN) $(TBFS_TARGET_BIN) $(MSSP_TARGET_BIN) $(SSWP_TARGET_BIN) $(ASSWP_TARGET_BIN) $(TSSWP_TARGET_BIN) $(KCORE_TARGET_BIN) $(CC_TARGET_BIN) $(ACC_TARGET_BIN) $(TCC_TARGET_BIN) $(CD_TARGET_BIN) $(ACD_TARGET_BIN) $(ABCD_TARGET_BIN) $(CDP2_TARGET_BIN) $(DEG_TARGET_BIN)

