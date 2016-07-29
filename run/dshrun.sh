#!/bin/bash

DSHFILE=$1;
CMD=$2;

dsh -M -f ${DSHFILE} -c ${CMD};
