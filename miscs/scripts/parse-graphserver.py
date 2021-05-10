import sys
import subprocess as sp
import boto3

lambda_base_price = 0.0000166667


def get_lambda_total_cost(filename):
    print("2. Calculating total lambda running time and cost...")
    lambda_name = ''
    start_time = 0
    end_time = -1
    with open(filename, 'r') as f:
        # Definitely need to think of how to do this better but it's fine for now
        for line in f:
            if 'Backend' in line:
                lambda_name = line.split()[6].split(':')[1]

            if 'Run start time' in line:
                start_time = line.split()[8:]
                start_time = " ".join(start_time)

            if 'Run end time' in line:
                end_time = line.split()[8:]
                end_time = " ".join(end_time)

    client = boto3.client('lambda')

    print("Lambda function name: " + lambda_name)
    print("Start time: " + start_time)
    print("End time: " + end_time)

    response = client.get_function_configuration(FunctionName=lambda_name)
    lambda_memory_size = float(response['MemorySize'])
    print("Lambda memory size: {} MB".format(lambda_memory_size))

    lambda_function_price = lambda_base_price / \
        (1024 / lambda_memory_size) / 1000

    logs_command = ["awslogs", "get", "/aws/lambda/", '--filter-pattern="Billed Duration"',
                    "--start='" + start_time + "'",
                    "--end='" + end_time + "'"]

    def get_logs(lambda_function):
        logs_command[2] += lambda_function
        print(" ".join(logs_command))
        output = sp.run(logs_command, capture_output=True, timeout=None)
        output = output.stdout.decode('utf-8')

        output = output.split('\n')
        output = [line.split() for line in output]
        times = [0]
        for words in output:
            try:
                index = words.index("Billed") + 2
                times.append(int(words[index]))
            except:
                continue

        return times

    times = get_logs(lambda_name)

    total_time = sum(times)
    print("Total time of lambdas for this run (ms):", total_time)
    print("Price of this lambda (based on memory):", lambda_function_price)
    print("Total cost of lambdas: ${:.3f}".format(
        total_time * lambda_function_price))


def get_server_runtime(filename):
    print("1. Calculate server running time...")
    sync_epochs = 0
    async_epochs = 0

    summed_sync_runtime, summed_async_runtime = 0., 0.
    max_sync_runtime, max_async_runtime = 0., 0.
    with open(filename, 'r') as f:
        for line in f:
            if 'sync epochs' in line and 'async epochs' in line:
                ''' The line looks like: "[ Node   0 ]  <EM>: 1 sync epochs and 99 async epochs"
                '''
                sync_epochs = int(line.split()[5])
                async_epochs = int(line.split()[9])

            if 'Average  sync epoch time' in line:
                sync_runtime = float(line.split()[-2])
                summed_sync_runtime += sync_runtime
                max_sync_runtime = max(max_sync_runtime, sync_runtime)

            if 'Average async epoch time' in line:
                async_runtime = float(line.split()[-2])
                summed_async_runtime += async_runtime
                max_async_runtime = max(max_async_runtime, async_runtime)

    training_time = max_sync_runtime * sync_epochs + max_async_runtime * async_epochs
    print("Training time in Table 3: {:.2f}s".format(training_time / 1000))
    # Now we have summed average sync/async epoch time of all servers. We will calculate the total summed graph server running time by multiplying them with number of epochs
    summed_sync_runtime *= sync_epochs
    summed_async_runtime *= async_epochs

    total_summed_runtime_in_second = (
        summed_sync_runtime + summed_async_runtime) / 1000
    print("Total summed graph server running time: {:.2f}s".format(
        total_summed_runtime_in_second))
    print("Look up the instance price on https://aws.amazon.com/ec2/pricing/on-demand/, and the server cost is the secondly price rate times summed running time.\n")


def main():
    if len(sys.argv) < 2:
        print('Usage: python {} <path to graphserver-out file>'.format(sys.argv[0]))
        exit(-1)
    filename = sys.argv[1]
    get_server_runtime(filename)
    get_lambda_total_cost(filename)


main()
