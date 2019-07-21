#!/bin/bash

#
# Run script. Make sure to set `dshmachines` and `zoo.basic` file in advance before
# running this script.
#
# Usage: ./ec2run.sh [Bench] [Dataset] [Feature-File]
#
#   Bench:          agg(*)
#   Dataset:        fb(*), small
#   Feature-File:   (*), path_to_file 
#
# "(*)" means default.
#


# Helper function for header displaying
function header {
  echo -e "\e[33;1m|---- ${1} ----> \e[0m"
}


#
# Preparations
#

user=$( whoami )

WORKDIR="/home/${user}"
OUTFILE_DIR="${WORKDIR}/outfiles"
ASPIREDIR="/home/${user}/aspire-streaming"
RUNDIR="/home/${user}/aspire-streaming/run"
DSHFILE="/home/${user}/aspire-streaming/run/dshmachines"
HOSTFILE="/home/${user}/aspire-streaming/run/hostfile"
TMPDIR="/home/${user}/zktmp"
ZOODIR=${WORKDIR}/aspire-streaming/installs/zookeeper-release-3.4.6

DSH=dsh

if [ ! -d ${OUTFILE_DIR} ]; then
	mkdir -p ${OUTFILE_DIR}
fi

if [ ! -f ${HOSTFILE} ]; then
	cat ${DSHFILE} | sed "s/${user}@//" > ${HOSTFILE}
fi

NDS=$(wc -l ${HOSTFILE} | cut -d" " -f1);   # NDS is number of distributed shells
ZOONDS=${NDS}

for i in $(seq 1 ${NDS}); do
  nodes[$i]=$(head -n $i ${HOSTFILE} | tail -n 1);
  dshnodes[$i]=$(head -n $i ${DSHFILE} | tail -n 1);
done;

header "Detected a cluster of ${NDS} nodes"


#
# Launch Zookeeper
#

header "Setting up tmp dir & Stopping running ZooKeeper..."

${DSH} -M -f ${DSHFILE} -c "rm -rf ${TMPDIR} && mkdir ${TMPDIR} && chown ${user}:${user} ${TMPDIR}"
${DSH} -M -f ${DSHFILE} -c "cd ${ZOODIR} && ./bin/zkServer.sh stop > /dev/null 2&>1";

if [ ! -f ${ZOODIR}/conf/zoo.cfg ]; then
	cat ${RUNDIR}/zoo.basic > ${ZOODIR}/conf/zoo.cfg;
	echo "" >> ${ZOODIR}/conf/zoo.cfg;
	for i in $(seq 1 ${ZOONDS}); do
	  echo "server.${i}=${nodes[$i]}:2080:3080" >> ${ZOODIR}/conf/zoo.cfg;
	done;
fi

header "Starting ZooKeeper..."

for i in $(seq 1 ${ZOONDS}); do
  scp ${ZOODIR}/conf/zoo.cfg ${dshnodes[$i]}:${ZOODIR}/conf/zoo.cfg;
  ${DSH} -M -m ${dshnodes[$i]} -c "mkdir -p ${TMPDIR}/zooDataDir";
  ${DSH} -M -m ${dshnodes[$i]} -c "echo $i > ${TMPDIR}/zooDataDir/myid";
  ${DSH} -M -m ${dshnodes[$i]} -c "cd ${ZOODIR} && ./bin/zkServer.sh start > /dev/null 2&>1";
done;

header "Checking for Quorum..."

for i in $(seq 1 ${ZOONDS}); do
  while true
  do
    str=$(echo stat | nc ${nodes[$i]} 2180 | grep "Mode");
    IFS=' ' read -ra ARR <<< ${str};
    if [[ ${ARR[1]} == "leader" ]] || [[ ${ARR[1]} == "follower" ]]; then
      break;
    fi
  done;
  echo "Found ZooKeeper node at ${nodes[$i]} -- ${str}";
done;


# 
# Do the work
#

header "Starting the benchmark..."

# Benchmark program
case $1 in 
	"agg")
		BM=aggregate.bin; BK=AGG;
		;;
	*)
		BM=aggregate.bin; BK=AGG;
		;;
esac

# Dataset
case $2 in
	"small")
		IP=../inputs/data/parts_${NDS}/small.graph.bsnap; IK=SM; SRC=0;
    ;;
	"fb")
		IP=/filepool/parts_${NDS}/facebook_combined.txt.bsnap; IK=FB; SRC=0
		;;
	*)
		IP=../inputs/data/parts_${NDS}/small.graph.bsnap; IK=SM; SRC=0;
		;;
esac

# Feature file
if [ -z $3 ]; then
  FF=$( dirname ${IP} )/../features;
else
  FF=$3;
fi

UD=0;
BE=100;
CT=7;
POF=1;
KC=10;
BS=100000;
TOA=0;
TOD=1;
STOD=1;
SP=1;
RS=0;

i=0
for i in $(seq 1 ${NDS}); do
  ${DSH} -M -m ${dshnodes[$i]} -c "echo ${nodes[$i]} > ${TMPDIR}/myip";
done;

rm ${ASPIREDIR}/config/hostfile;
rm ${ASPIREDIR}/config/zkhostfile;
echo -e "${nodes[1]}\tnode1\tmaster" > ${ASPIREDIR}/config/hostfile;
for i in $(seq 2 ${NDS}); do
  echo -e "${nodes[$i]}\tnode$i\tworker" >> ${ASPIREDIR}/config/hostfile;
done;

for i in $(seq 1 ${ZOONDS}); do
  echo -e "${nodes[$i]}\t2180" >> ${ASPIREDIR}/config/zkhostfile;
done;

for i in $(seq 1 ${NDS}); do
  scp ${ASPIREDIR}/config/hostfile ${ASPIREDIR}/config/zkhostfile ${dshnodes[$i]}:${ASPIREDIR}/config/;
done;

# Loop over desired number of runs
for dp in {1..1}; do

  cd ${RUNDIR};

  GVID=`cat gvid`;
  NGVID=$((GVID + 1));
  echo ${NGVID} > gvid;

  header "Running GVID #: ${GVID}"

  OPFILE=${OUTFILE_DIR}/${GVID}.${BK}.${IK}.out
  echo "GVID = ${GVID}" >> ${OPFILE} 2>&1;

  echo "DSH command (from ${ASPIREDIR}/build): ./${BM} --graphfile ${IP} --featuresfile ${FF} --undirected ${UD} --bm-reset=${RS} --bm-source=${SRC} --bm-tmpdir=${TMPDIR} --kcore-maxcore=${KC} --cthreads ${CT} ${XTRAARGS}";
  ${DSH} -M -f ${DSHFILE} -c "cd ${ASPIREDIR}/build && ./${BM} --graphfile ${IP} --featuresfile ${FF} --undirected ${UD} --bm-reset=${RS} --bm-source=${SRC} --bm-tmpdir=${TMPDIR} --kcore-maxcore=${KC} --cthreads ${CT} ${XTRAARGS}" >> ${OPFILE} 2>&1;

  DOPDIR=${ASPIREDIR}/build/outputs/${BK}.${IK}/${GVID};
  mkdir -p ${DOPDIR};
  for i in $(seq 1 ${NDS}); do
    oid=`expr $i - 1`;
    scp ${dshnodes[$i]}:${TMPDIR}/output_* ${DOPDIR}/;
  done;

done;


#
# Destroy ZooKeeper
#

header "Finished. Destroying ZooKeeper..."

for i in $(seq 1 $ZOONDS); do
  ${DSH} -M -m ${nodes[$i]} -c "cd ${ZOODIR} && ./bin/zkServer.sh stop > /dev/null 2&>1";
done;
