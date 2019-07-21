CPP=g++
INCPATH=../installs/out/include
LIBPATH=../installs/out/lib
CFLAGS=-O3

all: partitioner repartitioner tobinary

partitioner: partitioner.cpp
	${CPP} -I${INCPATH} -L${LIBPATH} $< -o $@ -lmetis ${CFLAGS}

tobinary: snapToBinarySnap.cpp
	${CPP} $< -o $@ ${CFLAGS}

.PHONY: clean
clean:
	rm -f partitioner tobinary
