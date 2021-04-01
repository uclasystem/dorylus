#!/bin/bash

cd $( dirname $0 )/../

EC2MAN_CMD='python3 -m ec2man'

$EC2MAN_CMD setup
USER=$( $EC2MAN_CMD graph user )
KEY=$( $EC2MAN_CMD graph key )

## Wait for all instances to be in state 'running' so
## that a PublicIpAddress will be allocated
for ctx in weight graph; do
	len=$( $EC2MAN_CMD $ctx all id | wc -l )
	for i in $(seq 0 $((len-1))); do
		instance_state=$( $EC2MAN_CMD $ctx $i state )
		echo "STATE: "$instance_state
		while [[ $instance_state != 'running' ]]; do
			if [[ $instance_state == 'stopped' ]]; then
				$EC2MAN_CMD $ctx $i start
			elif [[ $instance_state == 'terminated' ]]; then
				echo "An instance is terminated. Remove it from the list"
				exit 13
			fi
			sleep 5
			instance_state=$( $EC2MAN_CMD $ctx $i state )
			echo "STATE: "$instance_state
		done
	done
done

for ctx in weight graph; do
	$EC2MAN_CMD $ctx info
done

for ctx in weight graph; do
	IPS=$( $EC2MAN_CMD $ctx all pubip )
	echo $IPS
	for IP in $IPS; do
		ssh -i $KEY $USER@$IP ls
		while [[ $? -ne 0 ]]; do
			sleep 5
			ssh -i $KEY $USER@$IP ls
		done
		echo "Ready for SSH"
	done
done

echo "All 'graph' and 'weight' instances ready for SSH"
