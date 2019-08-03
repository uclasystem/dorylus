import argparse
import boto3
import json
import os
import pickle
import subprocess
    
import ec2_manager


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
EC2_DIR = BASE_DIR + "ec2_manager/"
CTX_DIR = EC2_DIR + "contexts/"


help_str = ("Usage: python3 -m ec2_manager help\n"
            "       python3 -m ec2_manager arn\n"
            "       python3 -m ec2_manager <Context> setup --user=<User> --key=<PrivateKeyFile> --tag=<AWSTag>\n"
            "       python3 -m ec2_manager <Context> info\n"
            "       python3 -m ec2_manager <Context> <NodeID> <Operation> [Args]\n"
            "       python3 -m ec2_manager <Context> all <Operation> [Args]\n"
            "\nOperations:\n"
            "\tid:\tGet the instance ID string of the node\n"
            "\tprip:\tGet the private ip address of the node\n"
            "\tpubip:\tGet the public ip address of the node\n"
            "\tssh:\tConnect to a node through SSH\n"
            "\tput:\tUse scp to put a file to the node's home directory\n"
            "\tget:\tUse scp to get a specific file from the node\n"
            "\tstart:\tStart the specified node\n"
            "\tstop:\tStop the specified node\n"
            "\treboot:\tRestart the current node\n"
            "\tstate:\tCheck the current state of the node\n"
            "\nTips:\n"
            "\tBefore using this tool, create an 'arn' file `ec2_manager/arn` which contains a line of your AWS ARN string.")


# Initialize the EC2 client.
ec2_cli = boto3.client('ec2')


# Read in the 'arn' file.
if not os.path.isfile(EC2_DIR + "arn"):
    print("ERROR: Not providing an `arn` file in the module directory. Please create one containing a line of your AWS ARN "
          "string and put it under `ec2_manager/` module path.")
    exit(1)

arn = ''
with open(EC2_DIR + "arn", 'r') as farn:
    arn = farn.readline().strip()


def parse_setup_args(args):
    """
    Parse the arguments for setting up a context.
    """

    parser = argparse.ArgumentParser(description="Process ec2 opts")
    parser.add_argument('--user', dest='user', type=str)
    parser.add_argument('--key', dest='key', type=str)
    parser.add_argument('--tag', dest='tag', type=str)

    return parser.parse_args(args)


def get_instances_info(tag):
    """
    Get instances information through the given tag.
    """

    from ec2_manager.classes import Instance

    quoted_tag = "'" + tag + "'"
    responses = ec2_cli.describe_instances(Filters=[{ 'Name': 'tag:Type', 'Values': [eval(quoted_tag)] }])

    instances = []
    for res in responses['Reservations']:
        for inst in reversed(res['Instances']):
            inst_id = inst['InstanceId']
            prip = inst['PrivateIpAddress']
            pubip = '0'
            if inst['State']['Name'] == 'running':
                pubip = inst['PublicIpAddress']
            instances.append(Instance(inst_id, prip, pubip))

    return instances


def process_options(ctx, target, args):
    """
    Process the given command arguments on the given context. Returns the resulting context in case
    it is changed.
    """

    from ec2_manager.command import handle_command

    # Process a single instance id.
    if (str.isdigit(target)):
        inst_id = int(target)
        if (inst_id >= len(ctx.instances)):
            show_error("No instance corresponding to the given index " + target + ".")
        else:
            ctx.instances[inst_id] = handle_command(ec2_cli, ctx, ctx.instances[inst_id], args)
    
    # Process all instances in the context.
    elif target == 'all':
        for i in range(len(ctx.instances)):
            print("Node " + str(i) + ":")
            ctx.instances[i] = handle_command(ec2_cli, ctx, ctx.instances[i], args)
            print()

    # Get information of current context.
    elif target == 'info':
        quoted_tag = "'" + ctx.tag + "'"
        response = ec2_cli.describe_instances(Filters=[{ 'Name': 'tag:Type', 'Values': [eval(quoted_tag)] }])

        running = 0
        for res in response['Reservations']:
            for inst in res['Instances']:
                if inst['State']['Name'] == 'running':
                    running += 1

        print("Context:", ctx.name)
        print("User:", ctx.user)
        print("Key:", ctx.key)
        print("Tag:", ctx.tag)
        print("Number of Instances: " + str(len(ctx.instances)) + ", where " + str(running) + " running")
    
    else:
        print("Option unrecognized. Use `python3 -m ec2_manager help` for the help message.")

    return ctx


def show_error(msg):
    """
    Shows an error message and displays the help string to users. Exit with error code 1.
    """

    print("ERROR: " + msg)
    print()
    print(help_str)
    exit(1)


def main(args):
    """
    Main entrance of this module. Basic usage: `python3 -m ec2_manager <GROUP> ...`.
    """

    from ec2_manager.classes import Context

    # Not providing a context, then must be querying arn / help message.
    if len(args) < 3:
        if len(args) == 2 and args[1] == "arn":
            print(arn)
            return
        elif len(args) == 2 and args[1] == "help":
            print(help_str)
            return
        else:
            show_error("Argument not recognized / Not enough arguments.")
    ctx_name, target = args[1], args[2]
    ctx_filename = CTX_DIR + ctx_name + ".context"

    # If doing `setup`, create the corresponding context. Other commands all require the previously created context.
    if target != "setup":
        if not os.path.isfile(ctx_filename):
            show_error("Context for '" + ctx_name + "' not found. Please run `setup` for it first.")
    else:
        if len(args) != 6:
            show_error("For setup, needs exactly '--user', '--key', and '--tag'.")
        else:
            if not os.path.isdir(CTX_DIR):
                os.mkdir(CTX_DIR)
            opts = parse_setup_args(args[3:])
            instances = get_instances_info(opts.tag)
            ctx = Context(ctx_name, opts.user, opts.key, opts.tag, instances)
            pickle.dump(ctx, open(ctx_filename, 'wb'))  # Dump the context to context file.
            print("Context for '" + ctx_name + "' created.")
            return
    
    # Context exists. Process the given command.
    ctx = pickle.load(open(ctx_filename, 'rb'))
    ctx = process_options(ctx, target, args[3:])        # The operation may change the context, so needs redump.
    pickle.dump(ctx, open(ctx_filename, 'wb'))
