#!/bin/bash

rm -rf graphlearn.*

DATASET=cora
JOB=
INDEX=
TRACKER=/filepool/tracker
while [[ $# -gt 0 ]]; do
	case "${1}" in
		-d|--dataset)
			DATASET="${2}"
			shift
			shift
			;;
		-j|--job)
			JOB="${2}"
			shift
			shift
			;;
		-i|--index)
			INDEX="${2}"
			shift
			shift
			;;
		*)
			shift
			;;
	esac
done

PY_CMD=train_supervised.py
RUN_CMD="python $PY_CMD \
  --ps_hosts=ip1:port1 \
  --worker_hosts=ip1:port2,ip2:port2,ip3:port2,... \
  --tracker=${TRACKER} \
  --dataset=${DATASET} \
  --job_name=${JOB} \
  --task_index=${INDEX}"

echo "Running command: $RUN_CMD"
$RUN_CMD
