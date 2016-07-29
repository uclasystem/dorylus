#!/bin/bash

if [ ! $# == 3 ]; then
    echo "Dude! Invoke like this: $0 <graph-file> <num-vertices> <num-partitions>";
    exit;
fi

GRAPH=$1;
VERTICES=$2;
PARTS=$3;

echo "Partitioning ...";
./partitioner ${GRAPH} ${VERTICES} ${PARTS};

#echo "Repartitioning ...";
#PFILE=parts_${PARTS}/${GRAPH}.parts;
#NEWPARTS=`expr ${PARTS} - 1`;
#./repartitioner ${GRAPH} ${PFILE} ${VERTICES} ${PARTS} ${NEWPARTS};

echo "Soft linking ...";
EFILE=parts_${PARTS}/${GRAPH}.edges;
GFILE=../${GRAPH};
ln -s ${GFILE} ${EFILE};
