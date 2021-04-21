#!/bin/bash

cd $( dirname $0 )

NWORKERS=
for var in "$@"; do
	if [[ $var = --nworkers=* ]]; then
		NWORKERS="${var#*=}"
	fi
done

echo $NWORKERS

./gnnman/setup-cluster $NWORKERS
./gnnman/mount-nfs-server &
python3 -m ec2man graph all ssh "mkdir -p feeds"
wait

./gnnman/send-source && ./gnnman/build-system graph gpu  &&  ./run/run-dorylus "$@"
