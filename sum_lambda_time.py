import subprocess as sp
import sys

logs_command = ["awslogs", "get", "/aws/lambda/",
                        "--start=" + sys.argv[-1]]

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
    forward_times = get_logs("forward")
    backward_times = get_logs("backward")

    all_times = forward_times + backward_times

    print("Number of forward lambdas counted:", len(forward_times))
    print("Number of backward lambdas counted:", len(backward_times))

    print("Total number of miliseconds billed by lambdas:", sum(all_times))
