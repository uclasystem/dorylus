TARGET      := myclient
TARGET_BIN  := $(TARGET).bin

#CPP         := g++ -std=c++11
CC         := gcc
RM          := rm -rf

# Flags
CFLAGS      := -DTHREADED -Wall -MMD -O3 ${EFLAGS}

#OMP_LIBS    := -fopenmp
#BOOST_LIBS  := -lboost_system -lboost_filesystem
ZK_LIBS     := -lzookeeper_mt
ZK_LIBPATH  := /home/keval/Desktop/workspace/installs/builds/zookeeper-cbinding/lib
LIBS        := -L$(ZK_LIBPATH) $(ZK_LIBS) #$(BOOST_LIBS) $(OMP_LIBS)

ZK_INCPATH  := /home/keval/Desktop/workspace/installs/builds/zookeeper-cbinding/include/zookeeper
INC         := -I$(ZK_INCPATH)

LDFLAGS     := -Wl,-rpath=$(ZK_LIBPATH)

OBJS        := myclient.o

#-include    $(PR_OBJS:%.o=%.d)

#all:        CFLAGS += -O3
all:        $(TARGET) 

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ $(INC) $(LIBS) $(LDFLAGS) $(INC) -o $(TARGET_BIN)

%.o: %.c %.h 
	$(CC) $(CFLAGS) $(INC) $(LIBS) $(LDFLAGS) $(INC) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INC) $(LIBS) $(LDFLAGS) $(INC) -c $< -o $@

clean:
	$(RM) *.o $(TARGET_BIN) *.d

