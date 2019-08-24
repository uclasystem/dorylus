import numpy as np


fcomp_results = {"fb":     np.empty(shape=(3, 15), dtype=float),
                 "small":  np.empty(shape=(3, 15), dtype=float),
                 "large":  np.empty(shape=(3, 15), dtype=float),
                 "reddit": np.empty(shape=(3, 15), dtype=float)}
fcomm_results = {"fb":     np.empty(shape=(3, 15), dtype=float),
                 "small":  np.empty(shape=(3, 15), dtype=float),
                 "large":  np.empty(shape=(3, 15), dtype=float),
                 "reddit": np.empty(shape=(3, 15), dtype=float)}
bcomp_results = {"fb":     np.empty(shape=(3, 15), dtype=float),
                 "small":  np.empty(shape=(3, 15), dtype=float),
                 "large":  np.empty(shape=(3, 15), dtype=float),
                 "reddit": np.empty(shape=(3, 15), dtype=float)}
bcomm_results = {"fb":     np.empty(shape=(3, 15), dtype=float),
                 "small":  np.empty(shape=(3, 15), dtype=float),
                 "large":  np.empty(shape=(3, 15), dtype=float),
                 "reddit": np.empty(shape=(3, 15), dtype=float)}


# Loop through all experiments.
run_mark = 0
for num_lambdas in range(1, 16):
    for dataset in ("fb", "small", "large", "reddit"):
        for run_no in range(3):

            run_mark += 1
            fres_name = "lambda/" + str(run_mark) + ".coord/output"

            # If line number incorrect, then invalid experiment. Ignore.
            if sum(1 for line in open(fres_name, 'r') \
                     if len(line.strip()) > 0) != 3 * 3 * num_lambdas:
                continue

            # Otherwise open the results file.
            count = 0
            with open(fres_name, 'r') as fres:

                # Loop through all lines.
                fcomp_timing, fcomm_timing, bcomp_timing, bcomm_timing = [0., 0.], [0., 0.], 0., 0.
                for line in fres.readlines():
                    line = line.strip()
                    if len(line) > 0:

                        # Is forward.
                        if 0 <= count < 3 * 2 * num_lambdas:
                            comm, _, comp, _ = tuple(map(float, line.split()[1:]))
                            fcomp_timing[count // (3 * num_lambdas)] += (comp / num_lambdas)
                            fcomm_timing[count // (3 * num_lambdas)] += (comm / num_lambdas)

                        # Is backward.
                        if 3 * 2 * num_lambdas <= count < 3 * 3 * num_lambdas:
                            comm, _, comp, _ = tuple(map(float, line.split()[1:]))
                            bcomp_timing += (comp / num_lambdas)
                            bcomm_timing += (comm / num_lambdas)

                        count += 1

                # Write the result into the results disctionary.
                fcomp_results[dataset][run_no, num_lambdas - 1] = sum(fcomp_timing)
                fcomm_results[dataset][run_no, num_lambdas - 1] = sum(fcomm_timing)
                bcomp_results[dataset][run_no, num_lambdas - 1] = bcomp_timing
                bcomm_results[dataset][run_no, num_lambdas - 1] = bcomm_timing


# Dump results.
with open("results", 'w+') as fout:
    fout.write("fcomp:\n")
    for dataset in ("fb", "small", "large", "reddit"):
        for i in range(3):
            for j in range(15):
                fout.write("{:.3f} ".format(fcomp_results[dataset][i, j]))
            fout.write("\n")
        fout.write("\n")
    fout.write("fcomm:\n")
    for dataset in ("fb", "small", "large", "reddit"):
        for i in range(3):
            for j in range(15):
                fout.write("{:.3f} ".format(fcomm_results[dataset][i, j]))
            fout.write("\n")
        fout.write("\n")
    fout.write("bcomp:\n")
    for dataset in ("fb", "small", "large", "reddit"):
        for i in range(3):
            for j in range(15):
                fout.write("{:.3f} ".format(bcomp_results[dataset][i, j]))
            fout.write("\n")
        fout.write("\n")
    fout.write("bcomm:\n")
    for dataset in ("fb", "small", "large", "reddit"):
        for i in range(3):
            for j in range(15):
                fout.write("{:.3f} ".format(bcomm_results[dataset][i, j]))
            fout.write("\n")
        fout.write("\n")
