import boto3
import subprocess
import time

import ec2man


def set_pub_ip(ec2_cli, instance):
    """
    Set public ip for an instance.
    """

    if instance.pub_ip == '0':  # Previously no pubip, so query again for now.
        response = ec2_cli.describe_instances(InstanceIds=[instance.id])['Reservations'][0]['Instances'][0]
        if response['State']['Name'] == "running" and 'PublicIpAddress' in response:
            instance.pub_ip = response['PublicIpAddress']

    return instance


def handle_command(ec2_cli, ctx, instance, op, args):
    """
    Handle a command to apply on the given instance.
    """

    # Set access options if ssh-key needed.
    remote_access_opts = []
    if instance.key != '':
        remote_access_opts = ["-i", instance.key]

    # Get instance id.
    if op == "id":
        print(instance.id)

    if op == "type":
        print(instance.type)

    # Get instance private ip.
    elif op == "prip":
        print(instance.pr_ip)

    # Query the current public ip, and return it.
    elif op == "pubip":
        instance = set_pub_ip(ec2_cli, instance)
        if instance.pub_ip == '0':
            print("Public IP address not yet defined. Make sure the instance is 'running'.")
        else:
            print(instance.pub_ip)

    # SSH connect to the instance.
    elif op == "ssh":
        instance = set_pub_ip(ec2_cli, instance)
        if instance.pub_ip == '0':
            print("Public IP address not yet defined. Make sure the instance is 'running'.")
        else:
            ssh_command = ['ssh'] + remote_access_opts + [instance.user + '@' + instance.pub_ip]
            if len(args) != 0:
                ssh_command += args
            subprocess.run(ssh_command)
    
    # Send file through scp to the instance.
    elif op == "put":
        instance = set_pub_ip(ec2_cli, instance)
        if instance.pub_ip == '0':
            print("Public IP address not yet defined. Make sure the instance is 'running'.")
        else:
            scp_command = ['scp'] + remote_access_opts
            if len(args) != 0:
                scp_command += args
                scp_command += [instance.user + '@' + instance.pub_ip + ':.']
                subprocess.run(scp_command)
            else:
                print("ERROR: No arguments given to operation `put`.")
                exit(1)

    # Send file through rsync to the instance.
    elif op == "rsync":
        instance = set_pub_ip(ec2_cli, instance)
        if instance.pub_ip == '0':
            print("Public IP address not yet defined. Make sure the instance is 'running'.")
        else:
            rsync_command = ["rsync", "-auzh", "-e"] + ["\"" + ' '.join(["ssh"] + remote_access_opts) + "\""]
            if len(args) != 0:
                rsync_command += args
                rsync_command += [instance.user + '@' + instance.pub_ip + ':.']
                rsync_command = ' '.join(rsync_command)
                subprocess.run(rsync_command, shell=True)
            else:
                print("ERROR: No arguments given to operation `rsync`.")
                exit(1)

    # Get file from the instance through scp.
    elif op == "get":
        instance = set_pub_ip(ec2_cli, instance)
        if instance.pub_ip == '0':
            print("Public IP address not yet defined. Make sure the instance is 'running'.")
        else:
            scp_command = ['scp'] + remote_access_opts
            if len(args) != 0:
                scp_command += [instance.user + '@' + instance.pub_ip + ':./' + args[0]]
                scp_command += ['.']
                subprocess.run(scp_command)
            else:
                print("ERROR: No arguments given to operation `get`.")
                exit(1)

    # Start the machine.
    elif op == "start":
        response = ec2_cli.start_instances(InstanceIds=[instance.id])
        print("Previous state:", response['StartingInstances'][0]['PreviousState']['Name'])
        print("Current state: ", response['StartingInstances'][0]['CurrentState']['Name'])

    # Stop the machine.
    elif op == "stop":
        instance.pub_ip = '0'   # Reset pubip to indicate "not running".
        response = ec2_cli.stop_instances(InstanceIds=[instance.id])
        print("Previous State:", response['StoppingInstances'][0]['PreviousState']['Name'])
        print("Current State: ", response['StoppingInstances'][0]['CurrentState']['Name'])

    # Reboot the instance.
    elif op == "reboot":
        print("Rebooting instance " + instance.id + "...")
        ec2_cli.reboot_instances(InstanceIds=[instance.id])

    # Check the state of the instance.
    elif op == "state":
        response = ec2_cli.describe_instances(InstanceIds=[instance.id])
        print(response['Reservations'][0]['Instances'][0]['State']['Name'])

    else:
        ec2man.show_error("Unrecognized operation `" + op + "`.")

    return instance
