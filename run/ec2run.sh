#!/bin/bash

WORKDIR="/home/keval/Desktop/workspace";
RUNDIR="/home/keval/Desktop/workspace/streaming/aspire/run";
DSHFILE="/home/keval/Desktop/workspace/streaming/aspire/run/dshmachines";
HOSTFILE="/home/keval/Desktop/workspace/streaming/aspire/run/hostfile";
TMPDIR="/home/keval/Desktop/workspace/streaming/aspire/run/tmp";
DSH=dsh;

cat ${DSHFILE} | sed 's/keval@//' > ${HOSTFILE};

NDS=$(wc -l ${HOSTFILE} | cut -d" " -f1);

for i in $(seq 1 ${NDS}); do
  nodes[$i]=$(head -n $i ${HOSTFILE} | tail -n 1);
  dshnodes[$i]=$(head -n $i ${DSHFILE} | tail -n 1);
done;

echo "Cluster of ${NDS} nodes";

echo "DSH Running: rm -rf ${TMPDIR} && mkdir ${TMPDIR} && chown keval:keval ${TMPDIR}";
${DSH} -M -f ${DSHFILE} -c "rm -rf ${TMPDIR} && mkdir ${TMPDIR} && chown keval:keval ${TMPDIR}";

############### INIT ZOOKEEPER ###############

ZOODIR=/home/keval/Desktop/workspace/installs/zookeeper-3.4.8;
ZOONDS=3;

echo "DSH Running: cd ${ZOODIR} && ./bin/zkServer.sh stop";
${DSH} -M -f ${DSHFILE} -c "cd ${ZOODIR} && ./bin/zkServer.sh stop";

cat ${RUNDIR}/zoo.basic > ${ZOODIR}/conf/zoo.cfg;
echo "" >> ${ZOODIR}/conf/zoo.cfg;
for i in $(seq 1 ${ZOONDS}); do
  echo "server.${i}=${nodes[$i]}:2080:3080" >> ${ZOODIR}/conf/zoo.cfg;
done;

echo "DSH Running: cd ${ZOODIR} && ./bin/zkServer.sh start";
for i in $(seq 1 ${ZOONDS}); do
  scp ${ZOODIR}/conf/zoo.cfg ${dshnodes[$i]}:${ZOODIR}/conf/zoo.cfg;
  ${DSH} -M -m ${dshnodes[$i]} -c "mkdir -p ${TMPDIR}/zooDataDir";
  ${DSH} -M -m ${dshnodes[$i]} -c "echo $i > ${TMPDIR}/zooDataDir/myid";
  ${DSH} -M -m ${dshnodes[$i]} -c "cd ${ZOODIR} && ./bin/zkServer.sh start";
done;

# CHECK FOR ZK QUORUM
for i in $(seq 1 ${ZOONDS}); do
  while true
  do
    str=$(echo stat | nc ${nodes[$i]} 2180 | grep "Mode");
    IFS=' ' read -ra ARR <<< ${str};
    if [[ ${ARR[1]} == "leader" ]] || [[ ${ARR[1]} == "follower" ]]; then
      break;
      echo "ZK node at ${nodes[$i]} -- ${str}";
    fi
  done;
  echo "ZK node at ${nodes[$i]} -- ${str}";
done;

############### DO WORK ###############

ASPIREDIR=/home/keval/Desktop/workspace/streaming/aspire;

UD=0;

#BM=pagerank.bin; BK=PR;
#BM=sssp.bin; BK=SSSP;
#BM=tsssp.bin; BK=TSSSP;
BM=sswp.bin; BK=SSWP;
#BM=asswp.bin; BK=ASSWP;
#BM=tsswp.bin; BK=TSSWP;
#BM=asssp.bin; BK=ASSSP;
#BM=connectedcomponents.bin; BK=CC; UD=1;
#BM=tconnectedcomponents.bin; BK=TCC; UD=1;
#BM=communitydetection.bin; BK=CD;
#BM=acommunitydetection.bin; BK=ACD;
#BM=abcommunitydetection.bin; BK=ABCD;
#BM=communitydetectionp2.bin; BK=CDP2;
#BM=kcore.bin; BK=KC;
#BM=degree.bin; BK=DG;

#IP=../inputs/parts_${NDS}/facebook_combined.txt.bsnap; IK=FB;
#IP=../inputs/parts_${NDS}/facebook_combined.txt_undir.bsnap; IK=FBU;
#IP=../inputs/parts_${NDS}/twitter_combined.txt.edited; IK=TT;
IP=../inputs/parts_${NDS}/soc-LiveJournal1.txt.bsnap; AF=NULL; IK=LJ; BE=300; SRC=10000;
#IP=../inputs/parts_${NDS}/soc-LiveJournal1.txt.bsnap.red.bsnap; AF=../inputs/parts_${NDS}/soc-LiveJournal1.txt.bsnap.red.bsnap.deleteadd; IK=SLJ;
#IP=../inputs/parts_${NDS}/roadNet-CA.txt.bsnap; AF=NULL; IK=RNCA; BE=30; SRC=0;

#IP=../inputs/parts_${NDS}/soc-LiveJournal1.txt_undir.bsnap; AF=NULL; IK=LJ; XTRAARGS="--cd-initfile=/home/keval/Desktop/workspace/aspire/inputs/cdp1outfinal";
#IP=../inputs/parts_${NDS}/soc-LiveJournal1.txt_undir.bsnap.red.bsnap; AF=../inputs/parts_${NDS}/soc-LiveJournal1.txt_undir.bsnap.red.bsnap.deleteadd; IK=SLJ;

OPFILE=${WORKDIR}/out.${BK}.${IK}.out;

CT=4;POF=1;
#SRC=10000;
KC=10;

NB=4;BS=100000;DP=10;
TOA=0;TOD=1;STOD=1;SP=1;
RS=1;

for i in $(seq 1 ${NDS}); do
  ${DSH} -M -m ${dshnodes[$i]} -c "echo ${nodes[$i]} > ${TMPDIR}/myip";
done;

rm -f ${OPFILE};
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
  echo -e "${nodes[$i]}\t2180" >> ${ASPIREDIR}/config/zkhostfile;
done;

for i in $(seq 1 ${NDS}); do
  scp ${ASPIREDIR}/config/hostfile ${ASPIREDIR}/config/zkhostfile ${dshnodes[$i]}:${ASPIREDIR}/config/;
done;

#-----

GVID=`cat gvid`
NGVID=$((GVID+1))
echo ${NGVID} > gvid;

echo "GVID = ${GVID}" >> ${OPFILE} 2>&1;

echo "DSH Running (from ${ASPIREDIR}/build): ./${BM} --graphfile ${IP} --undirected ${UD} --bm-reset=${RS} --bm-source=${SRC} --bm-tagonadd=${TOA} --bm-tagondelete=${TOD} --bm-smarttagondelete=${STOD} --bm-smartpropagation=${SP} --bm-tmpdir=${TMPDIR} --kcore-maxcore=${KC} --cthreads ${CT} --pofrequency ${POF} --baseedges ${BE} --numbatches ${NB} --batchsize ${BS} --deletepercent ${DP} ${XTRAARGS}";
${DSH} -M -f ${DSHFILE} -c "cd ${ASPIREDIR}/build && ./${BM} --graphfile ${IP} --undirected ${UD} --bm-reset=${RS} --bm-source=${SRC} --bm-tagonadd=${TOA} --bm-tagondelete=${TOD} --bm-smarttagondelete=${STOD} --bm-smartpropagation=${SP} --bm-tmpdir=${TMPDIR} --kcore-maxcore=${KC} --cthreads ${CT} --pofrequency ${POF} --baseedges ${BE} --numbatches ${NB} --batchsize ${BS} --deletepercent ${DP} ${XTRAARGS}" >> ${OPFILE} 2>&1;

DOPDIR=${ASPIREDIR}/build/outputs/${BK}.${IK}/${GVID};
mkdir -p ${DOPDIR};
for i in $(seq 1 ${NDS}); do
  #echo "${dshnodes[$i]}:${ASPIREDIR}/build/output_$i ${ASPIREDIR}/build/output_$i";
  oid=`expr $i - 1`;
  scp ${dshnodes[$i]}:${TMPDIR}/output_* ${DOPDIR}/;
  scp ${dshnodes[$i]}:${TMPDIR}/approx_* ${DOPDIR}/;
  #scp ${dshnodes[$i]}:${ASPIREDIR}/build/approx_${oid} ${ASPIREDIR}/build/approx_${oid};
done;

cd ${ASPIREDIR}/build && ./tester.sh ${DOPDIR}/ ${NB};

#-----

############### DESTROY ZOOKEEPER ###############

for i in $(seq 1 $ZOONDS); do
  ${DSH} -M -m ${nodes[$i]} -c "cd ${ZOODIR} && ./bin/zkServer.sh stop";
done;


