WEIGHTSERVER=ubuntu@3.86.80.110
COORDSERVER=ubuntu@3.84.126.224

cd ${HOME}
mkdir -p testres/lambda

dsh -m ${COORDSERVER} "./gnn-lambda/run/clear-out" > /dev/null 2>&1

RUN_MARK=1

for LAMBDAS in {1..15}; do

    echo ${LAMBDAS} > numlambdasforward
    echo ${LAMBDAS} > numlambdasbackward

    for DATASET in fb small large reddit; do
        for RUN in {1..3}; do
            echo "[ # ${RUN_MARK} ] Running ${RUN} of ${DATASET}"

            dsh -m ${WEIGHTSERVER} "./gnn-lambda/run/run-onnode weight" > /dev/null 2>&1 &
            dsh -m ${COORDSERVER} "./gnn-lambda/run/run-onnode coord" > /dev/null 2>&1 &
            
            sleep 1s
            timeout 90s ./gnn-lambda/run/run-onnode graph ${DATASET} > /dev/null 2>&1
            sleep 1s

            RUN_MARK=$(( ${RUN_MARK} + 1 ))
        done
    done
done

scp -r ${COORDSERVER}:outfiles/* testres/lambda/
