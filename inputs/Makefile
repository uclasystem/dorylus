CPP=g++
INCPATH=../installs/out/include
LIBPATH=../installs/out/lib
CFLAGS=-O3

all: partitioner tobinary tobinaryfeature

partitioner: partitioner.cpp
	${CPP} -I${INCPATH} -L${LIBPATH} $< -o $@ -lmetis ${CFLAGS}

tobinary: snapToBinarySnap.cpp
	${CPP} $< -o $@ ${CFLAGS}

tobinaryfeature: snapFeatures.cpp
	${CPP} $< -o $@ ${CFLAGS} -lboost_system

.PHONY: clean
clean:
	rm -f partitioner tobinary tobinaryfeature
