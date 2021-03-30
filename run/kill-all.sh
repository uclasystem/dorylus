#!/bin/bash

##
## Execute the kill-all command on graph server 0
## Make sure to run `./gnnman/setup-cluster` first so the kill-all script is
## properly configured
##

cd $( dirname $0 )/..
EC2MAN_CMD='python3 -m ec2man'

${EC2MAN_CMD} graph 0 ssh './kill-all'
