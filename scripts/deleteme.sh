#!/bin/bash

if [ ! $# -ge 3 ]; then
    echo "Dude! Invoke like this: $0 <benchmark> <input> <cfrequency> [<sssp-source>]";
    exit;
fi


BM=$1;
IP=$2;
CF=$3;

SRC=0;
if [ $# -ge 4 ]; then
    SRC=$4;
fi

cd /vagrant/project/basicft/utils;
./cleanzookeeper /fancyapp;

cd /vagrant/project/basicft/benchmarks;
echo "Running: ./${BM} --graphfile ${IP} --sssp-source=${SRC} --cfrequency=${CF}";
#./${BM} --graphfile ${IP} --sssp-source=${SRC} --cfrequency=${CF};
