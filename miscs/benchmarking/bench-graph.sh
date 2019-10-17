WEIGHTSERVER=ubuntu@3.86.80.110
COORDSERVER=ubuntu@3.84.126.224

cd ${HOME}
mkdir -p testres/graph

./gnn-lambda/run/clear-out > /dev/null 2>&1

for DATASET in fb small large reddit; do
    for RUN in 1 2 3; do
        echo
        echo "Running ${RUN} of ${DATASET}"
        echo

        dsh -m ${WEIGHTSERVER} "./gnn-lambda/run/run-onnode weight" > /dev/null 2>&1 &
        dsh -m ${COORDSERVER} "./gnn-lambda/run/run-onnode coord" > /dev/null 2>&1 &
        
        sleep 1s
        timeout 60s ./gnn-lambda/run/run-onnode graph ${DATASET}
        mv tmpfiles/output_0 testres/graph/${RUN}.${DATASET}
        sleep 1s
    done
done
