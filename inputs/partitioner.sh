#!/bin/bash

if [ ! $# == 3 ]; then
    echo "Usage: $0 <graph-file> <num-vertices> <num-partitions>";
    exit;
fi

GRAPH=$1;
VERTICES=$2;
PARTS=$3;

echo "Partitioning ...";
if [ ! -d data ]; then
    mkdir data
fi
./partitioner ${GRAPH} ${VERTICES} ${PARTS};
mv parts_${PARTS} data/

# echo "Soft linking ...";
# EFILE=parts_${PARTS}/${GRAPH}.edges;
# GFILE=../${GRAPH};
# ln -s ${GFILE} ${EFILE};
