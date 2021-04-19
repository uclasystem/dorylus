#!/bin/bash

cd $( dirname $0 )

NWORKERS=
for var in "$@"; do
	if [[ $var = --nworkers=* ]]; then
		NWORKERS="${var#*=}"
	fi
done

echo $NWORKERS

./gnnman/wait-for-ssh.sh
./gnnman/mount-nfs-server &
./gnnman/setup-cluster $NWORKERS
wait

./gnnman/send-source && ./gnnman/build-system graph gpu  &&  ./run/run-dorylus "$@"
