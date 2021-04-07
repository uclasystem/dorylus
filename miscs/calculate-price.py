import sys
import subprocess as sp
import boto3

lambda_base_price = 0.0000166667

filename = sys.argv[1]
lambda_name=''
start_time=0
end_time=-1
with open(filename, 'r') as f:
    ## Definitely need to think of how to do this better but it's fine for now
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

print(lambda_name)
print(start_time)
print(end_time)

response = client.get_function_configuration(FunctionName=lambda_name)
lambda_memory_size = float(response['MemorySize'])
print(lambda_memory_size)

lambda_function_price = lambda_base_price / (1024 / lambda_memory_size) / 1000

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
print(total_time * lambda_function_price)