#!/bin/bash

OP=${1}
CMD=
ARGS=
IP_FILE=~/group_ips
while [[ $# -gt 0 ]]; do
	case ${1} in
		-c|--command|-f|--files)
			ARGS="${2}"
			shift
			shift
			;;
		-ipf|--ip-file)
			IP_FILE="${2}"
			shift
			shift
			;;
		*)
			shift
			;;
	esac
done


for ip in $( cat $IP_FILE ); do
	case $OP in
		"do")
			ssh $ip "${ARGS}"
			;;
		"send")
			scp ${ARGS} $ip:$(pwd)/
			;;
		*)
			echo "Unknown op"
	esac
done
