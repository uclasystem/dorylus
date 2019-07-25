#!/bin/bash

#
# Run script. Make sure to set `dshmachines` and `zoo.basic` file in advance before
# running this script.
#
# Usage: ./ec2run.sh [Dataset] [Feature-File] [Layer-File]
#
#   Dataset:        small(*), fb
#   Feature-File:   (*), path_to_file 
#   Layer-File:     (*), path_to_file
#
# "(*)" means default.
#


# Helper function for header & result displaying
function header {
  echo -e "\e[33;1m${1}\e[0m"
}


#
# Preparations
#

user=$( whoami )

WORKDIR="/home/${user}"
LOGFILEDIR="${WORKDIR}/logfiles"
ASPIREDIR="/home/${user}/gnn-lambda"
RUNDIR="/home/${user}/gnn-lambda/run"
CONFIGDIR="/home/${user}/gnn-lambda/config"
DSHFILE="/home/${user}/gnn-lambda/run/dshmachines"
HOSTFILE="/home/${user}/gnn-lambda/run/hostfile"
TMPDIR="/home/${user}/zktmp"
ZOODIR=${WORKDIR}/gnn-lambda/installs/zookeeper-release-3.4.6

DSH=dsh

if [ ! -d ${LOGFILEDIR} ]; then
	mkdir -p ${LOGFILEDIR}
fi

if [ ! -d ${CONFIGDIR} ]; then
  mkdir -p ${CONFIGDIR}
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

header "Detected a cluster of ${NDS} nodes."


#
# Launch Zookeeper
#

header "Setting up tmp dir & Stopping running ZooKeeper..."

${DSH} -M -f ${DSHFILE} -c "rm -rf ${TMPDIR} && mkdir ${TMPDIR} && chown ${user}:${user} ${TMPDIR}";
${DSH} -M -f ${DSHFILE} -c "cd ${ZOODIR} && ./bin/zkServer.sh stop > /dev/null 2>&1";

if [ ! -f ${ZOODIR}/conf/zoo.cfg ]; then
	cat ${RUNDIR}/zoo.basic > ${ZOODIR}/conf/zoo.cfg;
	echo "" >> ${ZOODIR}/conf/zoo.cfg;
	for i in $(seq 1 ${ZOONDS}); do
	  echo "server.${i}=${nodes[$i]}:2080:3080" >> ${ZOODIR}/conf/zoo.cfg;
	done;
fi

header "Starting ZooKeeper..."

for i in $(seq 1 ${ZOONDS}); do
  scp -q ${ZOODIR}/conf/zoo.cfg ${dshnodes[$i]}:${ZOODIR}/conf/zoo.cfg;
  ${DSH} -M -m ${dshnodes[$i]} -c "mkdir -p ${TMPDIR}/zooDataDir";
  ${DSH} -M -m ${dshnodes[$i]} -c "echo $i > ${TMPDIR}/zooDataDir/myid";
  ${DSH} -M -m ${dshnodes[$i]} -c "cd ${ZOODIR} && ./bin/zkServer.sh start > /dev/null 2>&1";
done;

header "Checking for Quorum..."

sleep 1

count=0
limit=$(( NDS * 10 ))
for i in $(seq 1 ${ZOONDS}); do
  while true
  do
    ((count++))
    if [ ${count} -ge ${limit} ]; then
	    echo -e "\e[31;1m[ERROR] Could not establish quorum\e[0m"
	    exit
    fi
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

header "Starting the system..."

# Dataset
case $1 in
	"small")
		INPUT_LOC=../inputs/data/parts_${NDS}/small.graph.bsnap; IK=SM;
		;;
	"fb")
		INPUT_LOC=/filepool/parts_${NDS}/facebook_combined.txt.bsnap; IK=FB;
		;;
	*)
		INPUT_LOC=../inputs/data/parts_${NDS}/small.graph.bsnap; IK=SM;
		;;
esac

# Feature file
FEATUREFILE=$( dirname ${INPUT_LOC} )/../features;

# Layer configuration file
LAYERFILE=$( dirname ${INPUT_LOC} )/../layerconfig;

i=0
for i in $(seq 1 ${NDS}); do
  ${DSH} -M -m ${dshnodes[$i]} -c "echo ${nodes[$i]} > ${TMPDIR}/myip";
done;

rm -f ${CONFIGDIR}/hostfile;
rm -f ${CONFIGDIR}/zkhostfile;
echo -e "${nodes[1]}\tnode1\tmaster" > ${CONFIGDIR}/hostfile;
for i in $(seq 2 ${NDS}); do
  echo -e "${nodes[$i]}\tnode$i\tworker" >> ${CONFIGDIR}/hostfile;
done;

for i in $(seq 1 ${ZOONDS}); do
  echo -e "${nodes[$i]}\t2180" >> ${CONFIGDIR}/zkhostfile;
done;

for i in $(seq 1 ${NDS}); do
  scp -q ${CONFIGDIR}/hostfile ${CONFIGDIR}/zkhostfile ${dshnodes[$i]}:${CONFIGDIR}/;
done;

# Coordinate server info
COORDSERVER_CONF=${RUNDIR}/cserverinfo
CSERVER_IP=$( cat ${COORDSERVER_CONF} | awk '{print $1}' )
CSERVER_PORT=$( cat ${COORDSERVER_CONF} | awk '{print $2}' )

# Loop over desired number of runs
for dp in {1..1}; do

  UNDIRECTED=0;
  COMPUTATION_THREADS=7;
  DATACOMM_THREADS=1;

  cd ${RUNDIR};

  GVID=`cat gvid`;
  NGVID=$((GVID + 1));
  echo ${NGVID} > gvid;

  header "Running GVID [ # ${GVID} ]..."

  LOGFILE=${LOGFILEDIR}/${GVID}.${IK}.out
  echo "This is the log for round: GVID = ${GVID}" >> ${LOGFILE};

  echo "DSH command (from ${ASPIREDIR}/build): ./gnn-lambda.bin --graphfile ${INPUT_LOC} --featuresfile ${FEATUREFILE} --layerfile ${LAYERFILE} --coordserverip ${CSERVER_IP} --coordserverport ${CSERVER_PORT} --undirected ${UNDIRECTED} --tmpdir=${TMPDIR} --cthreads ${COMPUTATION_THREADS} --dthreads ${DATACOMM_THREADS}";
  ${DSH} -M -f ${DSHFILE} -c "cd ${ASPIREDIR}/build && ./gnn-lambda.bin --graphfile ${INPUT_LOC} --featuresfile ${FEATUREFILE} --layerfile ${LAYERFILE} --coordserverip ${CSERVER_IP} --coordserverport ${CSERVER_PORT} --undirected ${UNDIRECTED} --tmpdir=${TMPDIR} --cthreads ${COMPUTATION_THREADS} --dthreads ${DATACOMM_THREADS}" 1> /dev/null 2>> ${LOGFILE};

  DOPDIR=${ASPIREDIR}/build/outputs/${GVID}.${IK};
  mkdir -p ${DOPDIR};
  for i in $(seq 1 ${NDS}); do
    oid=`expr $i - 1`;
    scp -q ${dshnodes[$i]}:${TMPDIR}/output_* ${DOPDIR}/;
  done;

done;


#
# Destroy ZooKeeper
#

header "Finished. Destroying ZooKeeper..."

for i in $(seq 1 $ZOONDS); do
  ${DSH} -M -m ${nodes[$i]} -c "cd ${ZOODIR} && ./bin/zkServer.sh stop > /dev/null 2>&1";
done;

echo "Check the output files in \"build/outputs/\" folder."
echo "Check the running logs under \"~/logfiles/\" folder."

# Display the result
if [ -e ${DOPDIR}/output_0 ]; then
  echo -e "\e[92;1mThis round of execution seems successful, congrats ;)\e[0m"
else
  echo -e "\e[91;1mExecution fails (at least on this node), check the log file!\e[0m"
fi
