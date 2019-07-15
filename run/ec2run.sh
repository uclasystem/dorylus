#!/bin/bash

user=$( whoami )

WORKDIR="/home/${user}";
OUTFILE_DIR="${WORKDIR}/outfiles"
RUNDIR="/home/${user}/aspire-streaming/run";
DSHFILE="/home/${user}/aspire-streaming/run/dshmachines";
HOSTFILE="/home/${user}/aspire-streaming/run/hostfile";
TMPDIR="/home/${user}/zktmp"
DSH=dsh;

if [ ! -d ${OUTFILE_DIR} ]; then
	mkdir -p ${OUTFILE_DIR}
fi

if [ ! -f ${HOSTFILE} ]; then
	cat ${DSHFILE} | sed "s/${user}@//" > ${HOSTFILE}
fi

NDS=$(wc -l ${HOSTFILE} | cut -d" " -f1);

for i in $(seq 1 ${NDS}); do
  nodes[$i]=$(head -n $i ${HOSTFILE} | tail -n 1);
  dshnodes[$i]=$(head -n $i ${DSHFILE} | tail -n 1);
done;

echo "Cluster of ${NDS} nodes"

echo "DSH Running: rm -rf ${TMPDIR} && mkdir ${TMPDIR} && chown ${user}:${user} ${TMPDIR}"
${DSH} -M -f ${DSHFILE} -c "rm -rf ${TMPDIR} && mkdir ${TMPDIR} && chown ${user}:${user} ${TMPDIR}"

############### INIT ZOOKEEPER ###############

ZOODIR=${WORKDIR}/aspire-streaming/installs/zookeeper-release-3.4.6
ZOONDS=3

echo -e "\e[33;1mDSH Running: cd ${ZOODIR} && ./bin/zkServer.sh stop\e[0m";
${DSH} -M -f ${DSHFILE} -c "cd ${ZOODIR} && ./bin/zkServer.sh stop";
if [ ! -f ${ZOODIR}/conf/zoo.cfg ]; then
	cat ${RUNDIR}/zoo.basic > ${ZOODIR}/conf/zoo.cfg;
	echo "" >> ${ZOODIR}/conf/zoo.cfg;
	for i in $(seq 1 ${ZOONDS}); do
	  echo "server.${i}=${nodes[$i]}:2080:3080" >> ${ZOODIR}/conf/zoo.cfg;
	done;
fi

echo -e "\e[33;1mSTARTING ZOOKEEPER\e[0m";
echo -e "\e[33;1mDSH Running: cd ${ZOODIR} && ./bin/zkServer.sh start\e[0m";
for i in $(seq 1 ${ZOONDS}); do
  scp ${ZOODIR}/conf/zoo.cfg ${dshnodes[$i]}:${ZOODIR}/conf/zoo.cfg;
  ${DSH} -M -m ${dshnodes[$i]} -c "mkdir -p ${TMPDIR}/zooDataDir";
  ${DSH} -M -m ${dshnodes[$i]} -c "echo $i > ${TMPDIR}/zooDataDir/myid";
  ${DSH} -M -m ${dshnodes[$i]} -c "cd ${ZOODIR} && ./bin/zkServer.sh start";
done;


echo -e "\e[33;1mCHECKING FOR QUORUM NOW \e[0m"
# CHECK FOR ZK QUORUM
for i in $(seq 1 ${ZOONDS}); do
  while true
  do
    str=$(echo stat | nc ${nodes[$i]} 2188 | grep "Mode");
    IFS=' ' read -ra ARR <<< ${str};
    if [[ ${ARR[1]} == "leader" ]] || [[ ${ARR[1]} == "follower" ]]; then
      break;
      echo "ZK node at ${nodes[$i]} -- ${str}";
    fi
  done;
  echo "ZK node at ${nodes[$i]} -- ${str}";
done;


############### DO WORK ###############


ASPIREDIR=/home/${user}/aspire-streaming
echo -e "\e[33;1mSTARTING BENCHMARK\e[0m"

UD=0;

BM=aggregate.bin; BK=AGG;
case $1 in 
	"agg")
		BM=aggregate.bin; BK=AGG;
		;;
	"inc")
		BM=increment.bin; BK=INC;
		;;
	*)
		BM=aggregate.bin; BK=AGG;
		;;
esac

IP=/filepool/parts_${NDS}/facebook_combined.txt.bsnap; IK=FB; SRC=0
case $2 in
	"fb")
		IP=/filepool/parts_${NDS}/facebook_combined.txt.bsnap; IK=FB; SRC=0
		;;
	"data")
		IP=/filepool/parts_${NDS}/data.bsnap; IK=DT; SRC=0;
		;;
	*)
		IP=/filepool/parts_${NDS}/facebook_combined.txt.bsnap; IK=FB; SRC=0
		;;
esac

FF=$3 #features file


i=0

BE=100;
CT=7;POF=1;
#SRC=10000;
KC=10;

NB=11;BS=100000;DP=10;
TOA=0;TOD=1;STOD=1;SP=0;
RS=0;

for i in $(seq 1 ${NDS}); do
  ${DSH} -M -m ${dshnodes[$i]} -c "echo ${nodes[$i]} > ${TMPDIR}/myip";
done;

#rm -f ${OPFILE};
#rm -f ${ASPIREDIR}/build/output_*;
#rm -f ${ASPIREDIR}/build/outputs/${BK}.${IK}/*;
#rm -f ${ASPIREDIR}/build/approx_*;
rm ${ASPIREDIR}/config/hostfile;
rm ${ASPIREDIR}/config/zkhostfile;

echo -e "${nodes[1]}\tnode1\tmaster" > ${ASPIREDIR}/config/hostfile;
for i in $(seq 2 ${NDS}); do
  echo -e "${nodes[$i]}\tnode$i\tworker" >> ${ASPIREDIR}/config/hostfile;
done;

for i in $(seq 1 ${ZOONDS}); do
  echo -e "${nodes[$i]}\t2188" >> ${ASPIREDIR}/config/zkhostfile;
done;

for i in $(seq 1 ${NDS}); do
  scp ${ASPIREDIR}/config/hostfile ${ASPIREDIR}/config/zkhostfile ${dshnodes[$i]}:${ASPIREDIR}/config/;
done;

#-----

#BS=0;
#for bs in {1..10}; do
#  BS=$((BS + 100000));

#SP=1;
DP=30; SP=1; RS=0;
for dp in {1..1}; do
  #DP=$((DP + 10));

  NB=0; DP=0; BS=0;

  cd ${RUNDIR};

  GVID=`cat gvid`;
  NGVID=$((GVID + 1));
  echo ${NGVID} > gvid;
  OPFILE=${OUTFILE_DIR}/${GVID}.${BK}.${IK}.out

  echo "GVID = ${GVID}" >> ${OPFILE} 2>&1;
  echo "GVID = ${GVID}"

  echo "DSH Running (from ${ASPIREDIR}/build): ./${BM} --graphfile ${IP} --featuresfile ${3} --undirected ${UD} --bm-reset=${RS} --bm-source=${SRC} --bm-tagonadd=${TOA} --bm-tagondelete=${TOD} --bm-smarttagondelete=${STOD} --bm-smartpropagation=${SP} --bm-tmpdir=${TMPDIR} --kcore-maxcore=${KC} --cthreads ${CT} --pofrequency ${POF} --baseedges ${BE} --numbatches ${NB} --batchsize ${BS} --deletepercent ${DP} ${XTRAARGS}";

  ${DSH} -M -f ${DSHFILE} -c "cd ${ASPIREDIR}/build && ./${BM} --graphfile ${IP} --featuresfile ${3} --undirected ${UD} --bm-reset=${RS} --bm-source=${SRC} --bm-tagonadd=${TOA} --bm-tagondelete=${TOD} --bm-smarttagondelete=${STOD} --bm-smartpropagation=${SP} --bm-tmpdir=${TMPDIR} --kcore-maxcore=${KC} --cthreads ${CT} --pofrequency ${POF} --baseedges ${BE} --numbatches ${NB} --batchsize ${BS} --deletepercent ${DP} ${XTRAARGS}" >> ${OPFILE} 2>&1;

  DOPDIR=${ASPIREDIR}/build/outputs/${BK}.${IK}/${GVID};
  mkdir -p ${DOPDIR};
  for i in $(seq 1 ${NDS}); do
    #echo "${dshnodes[$i]}:${ASPIREDIR}/build/output_$i ${ASPIREDIR}/build/output_$i";
    oid=`expr $i - 1`;
    scp ${dshnodes[$i]}:${TMPDIR}/output_* ${DOPDIR}/;
    #scp ${dshnodes[$i]}:${TMPDIR}/approx_* ${DOPDIR}/;
    #scp ${dshnodes[$i]}:${ASPIREDIR}/build/approx_${oid} ${ASPIREDIR}/build/approx_${oid};
  done;

  cd ${ASPIREDIR}/build && ./tester.sh ${DOPDIR}/ ${NB};

done; # dp

#done;

#-----

############### DESTROY ZOOKEEPER ###############

for i in $(seq 1 $ZOONDS); do
  ${DSH} -M -m ${nodes[$i]} -c "cd ${ZOODIR} && ./bin/zkServer.sh stop";
done;

#sleep 10;
#cd /home/${user}/Desktop/workspace/aspire-streaming/run;
#/home/${user}/Desktop/workspace/aspire-streaming/run/ec3run.sh;

#/home/${user}/Desktop/workspace/management/commander.py stopworkers ;
