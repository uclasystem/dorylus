CPP=g++
INCPATH=../installs/out/include
LIBPATH=../installs/out/lib
CFLAGS=-O3

all: partitioner tobinary

partitioner: partitioner.cpp
	${CPP} -I${INCPATH} -L${LIBPATH} $< -o $@ -lmetis ${CFLAGS}

# repartitioner: repartitioner.cpp
# 	${CPP} -I${INCPATH} -L${LIBPATH} $< -o $@ -lmetis ${CFLAGS}

tobinary: snapToBinarySnap.cpp
	${CPP} $< -o snapToBinarySnap ${CFLAGS}

.PHONY: clean
clean:
	rm -f partitioner tobinary
