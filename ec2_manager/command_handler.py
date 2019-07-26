import boto3
import subprocess
import time

import ec2_manager

arn = "arn:aws:iam::754699445878:role/lambda-cpp-demo"

def check_pub_ip(ec2_cli, instance):
	if instance.pub_ip == '0':
		response = ec2_cli.describe_instances(InstanceIds=[instance.id])['Reservations'][0]['Instances'][0]
		if response['State']['Name'] == "running" and 'PublicIpAddress' in response:
			instance.pub_ip = response['PublicIpAddress']

	return instance


def handle_command(ec2_cli, ctx, instance, op, args):
	## Set access options if ssh-key needed
	remote_access_opts = []
	if ctx.key != '':
		remote_access_opts = ['-i', ctx.key]

	if op == "id":
		print(instance.id)
	elif op == "prip":
		print(instance.pr_ip)
	elif op == "reset":
		instance.pub_ip = '0'

	## Requires pub_ip to run: do check_pub_ip
	elif op == "pubip":
		instance = check_pub_ip(ec2_cli, instance)
		if instance.pub_ip == '0':
			print("Public IP address not yet defined. Make sure instance state is 'running'")
		else:
			print(instance.pub_ip)
	elif op == "ssh":
		instance = check_pub_ip(ec2_cli, instance)
		if instance.pub_ip == '0':
			print("Public IP address not yet defined. Make sure instance state is 'running'")
		else:
			ssh_command = ['ssh'] + remote_access_opts + [ctx.user + '@' + instance.pub_ip]
			if len(args) != 0:
				ssh_command += args

			print("Connecting to remote machine at " + ctx.user + "@" + instance.pub_ip)
			subprocess.run(ssh_command)
	elif op == "send":
		instance = check_pub_ip(ec2_cli, instance)
		if instance.pub_ip == '0':
			print("Public IP address not yet defined. Make sure instance state is 'running'")
		else:
			scp_command = ['scp'] + remote_access_opts
			if len(args) != 0:
				scp_command += args
				scp_command += [ctx.user + '@' + instance.pub_ip + ':.']
				subprocess.run(scp_command)
			else:
				print("No arguments given to 'send'")
	elif op == "get":
		instance = check_pub_ip(ec2_cli, instance)
		if instance.pub_ip == '0':
			print("Public IP address not yet defined. Make sure instance state is 'running'")
		else:
			scp_command = ['scp'] + remote_access_opts
			if len(args) == 1:
				scp_command += [ctx.user + '@' + instance.pub_ip + ':./' + args[0]]
				scp_command += ['.']
				subprocess.run(scp_command)
			else:
				print("No arguments given to 'get'")

	elif op == "start":
		response = ec2_cli.start_instances(InstanceIds=[instance.id])
		print("Previous State:", response['StartingInstances'][0]['PreviousState']['Name'])
		print("Current State:", response['StartingInstances'][0]['CurrentState']['Name'])
	elif op == "stop":
		instance.pub_ip = '0'
		response = ec2_cli.stop_instances(InstanceIds=[instance.id])
		print("Previous State:", response['StoppingInstances'][0]['PreviousState']['Name'])
		print("Current State:", response['StoppingInstances'][0]['CurrentState']['Name'])

	elif op == "reboot":
		ec2_cli.reboot_instances(InstanceIds=[instance.id])
		print("Rebooting instance")

	elif op == "check":
		response = ec2_cli.describe_instances(InstanceIds=[instance.id])
		print(response['Reservations'][0]['Instances'][0]['State']['Name'])
	else:
		print("Unrecognized op command")

	return instance
