import subprocess as sp
import sys

logs_command = []

mem_to_price = { "128"  :  0.0000002083,
                 "256"  :  0.0000002083 * 2,
                 "512"  :  0.0000008333,
                 "1024" :  0.0000016667,
                 "1536" :  0.0000025000 }

## Retrieve a list of times for either the forward or backward lambdas
## based on the `log_type` parameter
def get_logs(log_type):
    this_log_command = logs_command.copy()
    this_log_command[2] += log_type
    output = sp.run(this_log_command, stdout=sp.PIPE).stdout.decode('utf-8')

    output = output.split('\n')
    output = [line.split() for line in output if "Billed Duration" in line]

    times = [int(words[words.index("Billed") + 2]) for words in output]

    return times


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python sum_lambda_time.py <funcname> <memsize> <start> <end>")
        exit()

    if sys.argv[2] not in mem_to_price:
        print("Memory size parameter must be in", list(mem_to_price.keys()))
        exit()

    funcname = sys.argv[1]
    price = mem_to_price[sys.argv[2]]
    logs_command = ["awslogs", "get", "/aws/lambda/",
                    "--filter-pattern=Billed Duration",
                    "--start='" + sys.argv[-2] + "'",
                    "--end='" + sys.argv[-1] + "'"]

    forward_times = get_logs(funcname)
    backward_times = get_logs(funcname)

    all_times = forward_times + backward_times

    print("Number of forward lambdas counted:", len(forward_times))
    print("Number of backward lambdas counted:", len(backward_times))

    total_time = sum(all_times) / 100
    total_cost = total_time * price
    print("Total number of miliseconds billed by lambdas:", (sum(all_times) / 100))

    print("Total cost for this run:", total_cost);
