#!/bin/bash

cd $(dirname $0)

DATASET=${1}
NODE_INDEX=${2}
CNT=${3}
MARK=${4}


if [[ ${NODE_INDEX} -eq 0 ]]; then
	echo "RUNNING ps with DATASET=${DATASET} on node ${NODE_INDEX}"
	bash run-dist.sh -d ${DATASET} -j ps -i 0 &
fi

sleep 3

START=$(($NODE_INDEX * $CNT))
for i in $(seq 0 $((CNT - 1))); do
	WORKER_INDEX=$((${START} + $i))
	echo "RUNNING worker with DATASET=${DATASET} on ${NODE_INDEX} as INDEX=${WORKER_INDEX}"
	bash run-dist.sh -d ${DATASET} -j worker -i ${WORKER_INDEX} > worker${WORKER_INDEX}.output &
	sleep 5
done

wait
