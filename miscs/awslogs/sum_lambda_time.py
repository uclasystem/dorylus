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
    print(" ".join(this_log_command))
    output = sp.run(this_log_command, capture_output=True, timeout=None)
    print(output.stderr.decode('utf-8'))
    output = output.stdout.decode('utf-8')

    output = output.split('\n')
    output = [line.split() for line in output]

    print(len(output))
    print(output[0])
    times = [int(words[words.index("Billed") + 2]) for words in output]

    return times


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python sum_lambda_time.py <funcname> <memsize> <start> <end>")
        exit()

    if sys.argv[2] not in mem_to_price:
        print("Memory size parameter must be in", list(mem_to_price.keys()))
        exit()
    logs_command = ["awslogs", "get", "/aws/lambda/", '--filter-pattern="Billed Duration"',
                    "--start='" + sys.argv[-2] + "'",
                    "--end='" + sys.argv[-1] + "'"]

    times = get_logs("gcn")

    print("Number of lambdas counted:", len(times))

    print("Total number of miliseconds billed by lambdas:", (sum(times) / 100))
