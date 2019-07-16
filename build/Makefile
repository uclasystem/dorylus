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
INC_TARGET      := increment
PR_TARGET       := pagerank
SSSP_TARGET     := sssp
ASSSP_TARGET    := asssp
TSSSP_TARGET    := tsssp
BFS_TARGET      := bfs
ABFS_TARGET     := abfs
TBFS_TARGET     := tbfs
MSSP_TARGET     := mssp
SSWP_TARGET     := sswp
ASSWP_TARGET    := asswp
TSSWP_TARGET    := tsswp
KCORE_TARGET    := kcore
CC_TARGET       := connectedcomponents
ACC_TARGET      := aconnectedcomponents
TCC_TARGET      := tconnectedcomponents
CD_TARGET       := communitydetection
ACD_TARGET      := acommunitydetection
ABCD_TARGET     := abcommunitydetection
CDP2_TARGET     := communitydetectionp2
DEG_TARGET      := degree

AGG_TARGET_BIN    := $(AGG_TARGET).bin
INC_TARGET_BIN    := $(INC_TARGET).bin
PR_TARGET_BIN     := $(PR_TARGET).bin
SSSP_TARGET_BIN   := $(SSSP_TARGET).bin
ASSSP_TARGET_BIN  := $(ASSSP_TARGET).bin
TSSSP_TARGET_BIN  := $(TSSSP_TARGET).bin
BFS_TARGET_BIN    := $(BFS_TARGET).bin
ABFS_TARGET_BIN   := $(ABFS_TARGET).bin
TBFS_TARGET_BIN   := $(TBFS_TARGET).bin
MSSP_TARGET_BIN   := $(MSSP_TARGET).bin
SSWP_TARGET_BIN   := $(SSWP_TARGET).bin
ASSWP_TARGET_BIN  := $(ASSWP_TARGET).bin
TSSWP_TARGET_BIN  := $(TSSWP_TARGET).bin
KCORE_TARGET_BIN  := $(KCORE_TARGET).bin
CC_TARGET_BIN     := $(CC_TARGET).bin
TCC_TARGET_BIN    := $(TCC_TARGET).bin
ACC_TARGET_BIN    := $(ACC_TARGET).bin
CD_TARGET_BIN     := $(CD_TARGET).bin
ACD_TARGET_BIN    := $(ACD_TARGET).bin
ABCD_TARGET_BIN   := $(ABCD_TARGET).bin
CDP2_TARGET_BIN   := $(CDP2_TARGET).bin
DEG_TARGET_BIN    := $(DEG_TARGET).bin

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
INC_OBJS        := $(BASE_OBJS) $(OBJ_PATH)/$(INC_TARGET).o
PR_OBJS         := $(BASE_OBJS) $(OBJ_PATH)/$(PR_TARGET).o
SSSP_OBJS       := $(BASE_OBJS) $(OBJ_PATH)/$(SSSP_TARGET).o
ASSSP_OBJS      := $(BASE_OBJS) $(OBJ_PATH)/$(ASSSP_TARGET).o
TSSSP_OBJS      := $(BASE_OBJS) $(OBJ_PATH)/$(TSSSP_TARGET).o
BFS_OBJS        := $(BASE_OBJS) $(OBJ_PATH)/$(BFS_TARGET).o
ABFS_OBJS       := $(BASE_OBJS) $(OBJ_PATH)/$(ABFS_TARGET).o
TBFS_OBJS       := $(BASE_OBJS) $(OBJ_PATH)/$(TBFS_TARGET).o
MSSP_OBJS       := $(BASE_OBJS) $(OBJ_PATH)/$(MSSP_TARGET).o
SSWP_OBJS       := $(BASE_OBJS) $(OBJ_PATH)/$(SSWP_TARGET).o
ASSWP_OBJS      := $(BASE_OBJS) $(OBJ_PATH)/$(ASSWP_TARGET).o
TSSWP_OBJS      := $(BASE_OBJS) $(OBJ_PATH)/$(TSSSP_TARGET).o
KCORE_OBJS      := $(BASE_OBJS) $(OBJ_PATH)/$(KCORE_TARGET).o
CC_OBJS         := $(BASE_OBJS) $(OBJ_PATH)/$(CC_TARGET).o
TCC_OBJS        := $(BASE_OBJS) $(OBJ_PATH)/$(TCC_TARGET).o
ACC_OBJS        := $(BASE_OBJS) $(OBJ_PATH)/$(ACC_TARGET).o
CD_OBJS         := $(BASE_OBJS) $(OBJ_PATH)/$(CD_TARGET).o
ACD_OBJS        := $(BASE_OBJS) $(OBJ_PATH)/$(ACD_TARGET).o
ABCD_OBJS       := $(BASE_OBJS) $(OBJ_PATH)/$(ABCD_TARGET).o
CDP2_OBJS       := $(BASE_OBJS) $(OBJ_PATH)/$(CDP2_TARGET).o
DEG_OBJS        := $(BASE_OBJS) $(OBJ_PATH)/$(DEG_TARGET).o


#
# MAKE targets
#

# Compiler & Flags
CPP     := g++ -std=c++11
CFLAGS  := -DTHREADED -Wall -Wno-sign-compare -MMD
ifeq ($(mode), release)
	CFLAGS += -O3
else
	CFLAGS += -DVERBOSE_ERRORS -g -fsanitize=address
endif

# Overall batch target
all: prompt directories agg_tgt inc_tgt pr_tgt sssp_tgt asssp_tgt tsssp_tgt bfs_tgt abfs_tgt tbfs_tgt mssp_tgt sswp_tgt asswp_tgt tsswp_tgt kcore_tgt cc_tgt tcc_tgt cd_tgt acd_tgt abcd_tgt cdp2_tgt deg_tgt 

# Benchmark individual targets
$(AGG_TARGET):      prompt directories agg_tgt
$(INC_TARGET):      prompt directories inc_tgt
$(PR_TARGET):       prompt directories pr_tgt
$(SSSP_TARGET):     prompt directories sssp_tgt
$(ASSSP_TARGET):    prompt directories asssp_tgt
$(TSSSP_TARGET):    prompt directories tsssp_tgt
$(BFS_TARGET):      prompt directories bfs_tgt
$(ABFS_TARGET):     prompt directories abfs_tgt
$(TBFS_TARGET):     prompt directories tbfs_tgt
$(MSSP_TARGET):     prompt directories mssp_tgt
$(SSWP_TARGET):     prompt directories sswp_tgt
$(ASSWP_TARGET):    prompt directories asswp_tgt
$(TSSWP_TARGET):    prompt directories tsswp_tgt
$(KCORE_TARGET):    prompt directories kcore_tgt
$(CC_TARGET):       prompt directories cc_tgt
$(TCC_TARGET):      prompt directories tcc_tgt
$(ACC_TARGET):      prompt directories acc_tgt
$(CD_TARGET):       prompt directories cd_tgt
$(ACD_TARGET):      prompt directories acd_tgt
$(ABCD_TARGET):     prompt directories abcd_tgt
$(CDP2_TARGET):     prompt directories cdp2_tgt
$(DEG_TARGET):      prompt directories deg_tgt

# Phony utilities
prompt:
	ifneq ($(mode), release)
		ifneq ($(mode), debug)
			@echo "Invalid build mode. Please specify either `mode=release` or `mode=debug`." 
			@exit 1
		endif
	endif
	@echo "Building on "$(mode)" mode....."

directories:
	mkdir -p $(OBJ_PATH) 

# Detailed target rules
agg_tgt: $(AGG_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFALGS) -o $(AGG_TARGET_BIN)

inc_tgt: $(INC_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(INC_TARGET_BIN)

pr_tgt: $(PR_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(PR_TARGET_BIN)

sssp_tgt: $(SSSP_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(SSSP_TARGET_BIN)

asssp_tgt: $(ASSSP_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(ASSSP_TARGET_BIN)

tsssp_tgt: $(TSSSP_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(TSSSP_TARGET_BIN)

bfs_tgt: $(BFS_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(BFS_TARGET_BIN)

abfs_tgt: $(ABFS_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(ABFS_TARGET_BIN)

tbfs_tgt: $(TBFS_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(TBFS_TARGET_BIN)

mssp_tgt: $(MSSP_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(MSSP_TARGET_BIN)

sswp_tgt: $(SSWP_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(SSWP_TARGET_BIN)

asswp_tgt: $(ASSWP_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(ASSWP_TARGET_BIN)

tsswp_tgt: $(TSSWP_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(TSSWP_TARGET_BIN)

kcore_tgt: $(KCORE_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(KCORE_TARGET_BIN)

cc_tgt: $(CC_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(CC_TARGET_BIN)

acc_tgt: $(ACC_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(ACC_TARGET_BIN)

tcc_tgt: $(TCC_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(TCC_TARGET_BIN)

cd_tgt: $(CD_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(CD_TARGET_BIN)

acd_tgt: $(ACD_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(ACD_TARGET_BIN)

abcd_tgt: $(ABCD_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(ABCD_TARGET_BIN)

cdp2_tgt: $(CDP2_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(CDP2_TARGET_BIN)

deg_tgt: $(DEG_OBJS)
	$(CPP) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) -o $(DEG_TARGET_BIN)

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
	rm -rf $(OBJ_PATH) $(AGG_TARGET_BIN) $(INC_TARGET_BIN) $(PR_TARGET_BIN) $(SSSP_TARGET_BIN) $(ASSSP_TARGET_BIN) $(TSSSP_TARGET_BIN) $(BFS_TARGET_BIN) $(ABFS_TARGET_BIN) $(TBFS_TARGET_BIN) $(MSSP_TARGET_BIN) $(SSWP_TARGET_BIN) $(ASSWP_TARGET_BIN) $(TSSWP_TARGET_BIN) $(KCORE_TARGET_BIN) $(CC_TARGET_BIN) $(ACC_TARGET_BIN) $(TCC_TARGET_BIN) $(CD_TARGET_BIN) $(ACD_TARGET_BIN) $(ABCD_TARGET_BIN) $(CDP2_TARGET_BIN) $(DEG_TARGET_BIN)

