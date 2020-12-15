#!/bin/bash

dataset=${1}
cnt=${2}
cnt=${cnt:-1}
host_file=${3}
host_file=${host_file:-~/group_ips}

TRACKER=/filepool/tracker
echo "Resetting tracker directory ${TRACKER}"
rm -rf ${TRACKER}
mkdir -p ${TRACKER}

run_mark=0
if [[ -f run-mark ]]; then
	run_mark=$( cat run-mark )
fi
echo $(( $run_mark + 1 )) > run-mark

index=0
for ip in $( cat ${host_file} ); do
	ssh $ip "cd graph-learn/examples/tf/gcn; bash deploy-2.sh ${dataset} ${index} ${cnt} ${run_mark}" &
	((index++))
	sleep 2
done

wait
