WEIGHTSERVER=ubuntu@54.210.93.127
COORDSERVER=ubuntu@54.235.47.102

cd ${HOME}
mkdir -p testres

for DATASET in fb small large reddit; do
	for RUN in 1 2 3; do
		echo
		echo "Running ${RUN} of ${DATASET}"
		echo
		dsh -m ${WEIGHTSERVER} "./gnn-lambda/run/run-onnode weight" > /dev/null 2>&1 &
		dsh -m ${COORDSERVER} "./gnn-lambda/run/run-onnode coord" > /dev/null 2>&1 &
		sleep 1s
		timeout 60s ./gnn-lambda/run/run-onnode graph ${DATASET}
		mv tmpfiles/output_0 testres/${RUN}.${DATASET}
		sleep 1s
	done
done
