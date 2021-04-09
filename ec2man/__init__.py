import boto3
import json
import os
import sys
import pickle
import subprocess
import multiprocessing
import argparse

import ec2man
import ec2man.instance_manager


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
EC2_DIR = BASE_DIR + "ec2man/"
CTX_DIR = EC2_DIR + "contexts/"


help_str = ("Usage: python3 -m ec2man help\n"
            "       python3 -m ec2man allocate [--ami AMI] [--type TYPE] [--cnt CNT] [--ctx CTX] [--az AZ] [--sg SG]\n"
            "       python3 -m ec2man add <Context> <ec2 id> [ec2 id]...\n"
            "       python3 -m ec2man setup\n"
            "       python3 -m ec2man <Context> info\n"
            "       python3 -m ec2man <Context> <NodeID> <Operation> [Args]\n"
            "       python3 -m ec2man <Context> all <Operation> [Args]\n"
            "\nOperations:\n"
            "\tid:\tGet the instance ID string of the node\n"
            "\ttype:\tCheck the type of the instance\n"
            "\tprip:\tGet the private ip address of the node\n"
            "\tpubip:\tGet the public ip address of the node\n"
            "\tssh:\tConnect to a node through SSH; Can append with command\n"
            "\tput:\tUse scp to put a file to the node's home directory\n"
            "\trsync:\tUse rsync to efficiently put a file to the node's home directory\n"
            "\tget:\tUse scp to get a specific file from the node\n"
            "\tstart:\tStart the specified node\n"
            "\tstop:\tStop the specified node\n"
            "\treboot:\tRestart the current node\n"
            "\tstate:\tCheck the current state of the node\n"
            "\nTips:\n"
            "\tBefore using this tool, create a 'profile' file `ec2man/profile` which contains a line of your AWS ARN string "
            "and a line of profile name to use. Also create a 'machines' file `ec2man/machines` which contains all the infos "
            "needed for setting up your instances.")


# Read in the profile info.
if not os.path.isfile(EC2_DIR + "profile"):
    print("ERROR: Not providing a `profile` file in the module directory.")
    exit(1)

profile_name = ''
user_name = ''
region = 'us-east-2'
with open(EC2_DIR + "profile", 'r') as fprofile:
    profile_name = fprofile.readline().strip().split()[0]
    if profile_name.lower() == 'default':   # If given as default, then set to None so that boto3 session uses the default profile.
        profile_name = None

    user_name = fprofile.readline().strip().split()[0]
    ssh_key = fprofile.readline().strip().split()[0]
    region = fprofile.readline().strip()


# Initialize the EC2 client.
boto3.setup_default_session(profile_name=profile_name)
ec2_cli = boto3.client('ec2', region_name=region)


def process_setup(machine_list = 'machines'):
    """
    Process the arguments for setting up the machines.
    """
    from ec2man.classes import Context

    if not os.path.isfile(EC2_DIR + machine_list):
        show_error("Config file '" + EC2_DIR + "machines' not found.")

    # Read through the config file, add all instances into corresponding context's id_list.
    contexts = dict()
    with open(EC2_DIR + machine_list, 'r') as fconfig:
        for line in fconfig.readlines():
            line = line.strip()
            if len(line) > 0:
                inst_id, role = tuple(line.split()[:2])
                if role not in contexts:
                    contexts[role] = dict()
                    print("Context for '" + role + "' created.")
                contexts[role][inst_id] = (user_name, ssh_key)

    # For every context, get its instances' info through its id_list. Create the context object, and dump
    # into a binary context file.
    for role in contexts:
        instances = get_instances_info(list(contexts[role].keys()))
        for inst in instances:
            inst.set_user_key(contexts[role][inst.id])
        ctx = Context(role, instances)
        pickle.dump(ctx, open(CTX_DIR + role + ".context", 'wb'))


def get_instances_info(id_list):
    """
    Get instances information through the given instance id.
    """
    from ec2man.classes import Instance

    responses = ec2_cli.describe_instances(InstanceIds=id_list)

    instances = []
    for res in responses['Reservations']:
        for inst in reversed(res['Instances']):
            inst_id = inst['InstanceId']
            inst_type = inst['InstanceType']
            inst_place = inst['Placement']['AvailabilityZone']
            prip = inst['PrivateIpAddress']
            pubip = '0'
            if inst['State']['Name'] == 'running':
                pubip = inst['PublicIpAddress']
            instances.append(Instance(inst_id, inst_type, inst_place, prip, pubip))

    instances = sorted(instances, key=lambda x: x.id)
    return instances


def process_target(ctx, target, args):
    """
    Process the given command arguments on the given context. Returns the resulting context in case
    it is changed.
    """
    from ec2man.command import handle_command

    # Process a single instance id.
    if (str.isdigit(target)):
        idx = int(target)
        if (idx >= len(ctx.instances)):
            show_error("No instance corresponding to the given index " + target + ".")
        if len(args) < 1:
            show_error("Please specify an operation.")
        op, args = args[0], args[1:]
        ctx.instances[idx] = handle_command(ec2_cli, ctx, ctx.instances[idx], op, args)

    # Process all instances in the context.
    elif target == 'all':
        if len(args) < 1:
            show_error("Please specify an operation.")
        op, args = args[0], args[1:]

        if op == "put" or op == "rsync":    # Multithreading these ops.

            threads = []

            def thread_handler(tid):
                ctx.instances[tid] = handle_command(ec2_cli, ctx, ctx.instances[tid], op, args)

            for i in range(len(ctx.instances)):
                t = multiprocessing.Process(target=thread_handler, args=(i,))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

        else:                               # Normal single threaded.
            for i in range(len(ctx.instances)):
                ctx.instances[i] = handle_command(ec2_cli, ctx, ctx.instances[i], op, args)

    # Get information of current context.
    elif target == 'info':
        print("Context '" + ctx.name + "' has " + str(len(ctx.instances)) + " instances:")
        for i in range(len(ctx.instances)):
            inst = ctx.instances[i]
            response = ec2_cli.describe_instances(InstanceIds=[inst.id])
            state = response['Reservations'][0]['Instances'][0]['State']['Name']
            print("  {:2d}  {}  {}  {}  {:15}  {}".format(i, inst.id, inst.type, inst.placement, inst.pr_ip, state))

    elif target == 'user':
        print(user_name)

    elif target == 'key':
        print(ssh_key)

    # Dump the dshmachines file for the given context.
    elif target == 'dshfile':
        for inst in ctx.instances:
            print(inst.user + "@" + inst.pr_ip)

    elif target == 'workersfile':
        workers_per_node = int(args[0])
        for inst in ctx.instances:
            for local_id in range(workers_per_node):
                print(inst.user + "@" + inst.pr_ip + ":" + str(local_id))

    else:
        print("Option unrecognized. Use `python3 -m ec2man help` for the help message.")

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
    Main entrance of this module. Basic usage: `python3 -m ec2man <GROUP> ...`.
    """

    from ec2man.classes import Context

    # Not providing a context, then must be querying help message or doing setup.
    if len(args) == 2 and args[1] == "help":
        print(help_str)
        return
    elif len(args) >= 2 and args[1] == "setup":
        if not os.path.isdir(CTX_DIR):
            os.mkdir(CTX_DIR)
        machine_list = "machines"
        if len(args) >= 3:
            machine_list = args[2]
        process_setup(machine_list)
        return

    elif len(args) >= 2 and args[1] == 'allocate':
        instance_manager.launch_ec2_instances(sys.argv[2:])
        return

    elif len(args) >= 2 and args[1] == 'add':
        instance_manager.add_managed_ec2_instance(sys.argv[2:])
        return

    ctx_name, target = args[1], args[2]
    ctx_filename = CTX_DIR + ctx_name + ".context"

    # Commands all require a corresponding context.
    if not os.path.isfile(ctx_filename):
        show_error("Context for '" + ctx_name + "' not found. Please do proper `setup` first.")

    # Context exists. Process the given command.
    ctx = pickle.load(open(ctx_filename, 'rb'))
    ctx = process_target(ctx, target, args[3:])     # The operation may change the context, so needs redump.
    pickle.dump(ctx, open(ctx_filename, 'wb'))
