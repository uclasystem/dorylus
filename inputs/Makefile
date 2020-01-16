CPP=g++
CFLAGS=-std=c++11 -O3


all: partitioner graphtobinary featurestobinary labelstobinary

convert2csc: convert2csc.cpp
	$(CPP) $< -o $@ ${CFLAGS}
partitioner: partitioner.cpp
	${CPP} $< -o $@ ${CFLAGS} -lmetis

graphtobinary: graphToBinary.cpp
	${CPP} $< -o $@ ${CFLAGS}

featurestobinary: featuresToBinary.cpp
	${CPP} $< -o $@ ${CFLAGS} -lboost_system

labelstobinary: labelsToBinary.cpp
	${CPP} $< -o $@ ${CFLAGS} -lboost_system


.PHONY: clean
clean:
	rm -f partitioner graphtobinary featurestobinary labelstobinary
