CPP=g++
INCPATH=../installs/out/include
LIBPATH=../installs/out/lib
CFLAGS=-std=c++11 -O3

all: partitioner graphtobinary featurestobinary

partitioner: partitioner.cpp
	${CPP} -I${INCPATH} -L${LIBPATH} $< -o $@ -lmetis ${CFLAGS}

graphtobinary: graphToBinary.cpp
	${CPP} $< -o $@ ${CFLAGS}

featurestobinary: featuresToBinary.cpp
	${CPP} $< -o $@ ${CFLAGS} -lboost_system

.PHONY: clean
clean:
	rm -f partitioner graphtobinary featurestobinary
