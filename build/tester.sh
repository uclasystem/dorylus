#/bin/bash

DIR=$1;
BS=$2;

for i in $(seq 1 ${BS}); do
  rm -f tmp;
  rm -f atmp;

  for f in ${DIR}/output_${i}_*; do
    cat $f >> tmp;
  done;

  for f in ${DIR}/approx_${i}_*; do
   cat $f >> atmp;
  done;

  sort -g tmp > ${DIR}/outfinal_${i};
#  sort -g atmp > ${DIR}/approxfinal_${i};

  rm tmp; 
  rm atmp;

  #for f in ./output_${i}_*; do
  #  rm $f;
  #done;

  #for f in ./approx_${i}_*; do
  # rm $f;
  #done;

done;

exit;

#diff outfinal op/outfinal > /dev/null; 
#rc=$?;

#if [ $rc == 0 ]; then
#    echo "TESTCASE PASSED!";
#else
#    echo "TESTCASE FAILED with return code $rc!";
#fi
