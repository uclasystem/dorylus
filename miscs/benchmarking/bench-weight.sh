WEIGHTSERVER=ubuntu@3.86.80.110
COORDSERVER=ubuntu@3.84.126.224

cd ${HOME}
mkdir -p testres/weight

dsh -m ${WEIGHTSERVER} "./gnn-lambda/run/clear-out" > /dev/null 2>&1

for RUN in 1 2 3; do
    echo
    echo "Running ${RUN} of fb"
    echo

    dsh -m ${WEIGHTSERVER} "./gnn-lambda/run/run-onnode weight" > /dev/null 2>&1 &
    dsh -m ${COORDSERVER} "./gnn-lambda/run/run-onnode coord" > /dev/null 2>&1 &
    
    sleep 1s
    timeout 60s ./gnn-lambda/run/run-onnode graph fb
    sleep 1s
done

scp -r ${WEIGHTSERVER}:outfiles/* testres/weight/
