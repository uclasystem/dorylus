import argparse
import boto3
import json
import os
import pickle
import subprocess
	
import ec2_manager
from ec2_manager.instance_info import Context

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EC2_DIR = BASE_DIR + "/ec2_manager/"
RES_DIR = EC2_DIR + "res/"

ec2_cli = boto3.client('ec2')

def get_instances_info(tag):
	from ec2_manager.instance_info import Instance
	quoted_tag = "'" + tag + "'"
	print("Getting instance info for " + quoted_tag)
	response = ec2_cli.describe_instances(Filters=[{ 'Name': 'tag:Type', 'Values': [eval(quoted_tag)] }])

	instances = []
	for res in response['Reservations']:
		for inst in reversed(res['Instances']):
			inst_id = inst['InstanceId']
			prip = inst['PrivateIpAddress']
			pubip = '0'
			if inst['State']['Name'] == 'running':
				pubip = inst['PublicIpAddress']

			instances.append(Instance(inst_id, prip, pubip))
	return instances


def parse_setup_opts(args):
	parser = argparse.ArgumentParser(description="Process ec2 opts")
	parser.add_argument('--user', dest='user', type=str)
	parser.add_argument('--key', dest='key', type=str)
	parser.add_argument('--tag', dest='tag', type=str)

	return parser.parse_args(args)


def process_options(ctx, args):
	from ec2_manager.command_handler import handle_command
	if (str.isdigit(args[1])):
		inst_id = int(args[1])
		if (inst_id >= len(ctx.instances)):
			print("No instance corresponding to id")
		else:
			ctx.instances[inst_id] = handle_command(ec2_cli, ctx, ctx.instances[inst_id], args[2], args[3:])
	else:
		if args[1] == 'setup':
			if len(args) < 5:
				print("For setup, need '--user' '--key' and '--tag'")
				return
			else:
				opts = parse_setup_opts(args[2:])
				instances = get_instances_info(opts.tag)
				if (len(instances)) == 0:
					print("No instances found with that tag")
				else:
					ctx = Context(opts.user, opts.key, opts.tag, instances)
		elif args[1] == 'all':
			for i in range(len(ctx.instances)):
				print("Node " + str(i) + ":")
				ctx.instances[i] = handle_command(ec2_cli, ctx, ctx.instances[i], args[2], args[3:])
				print()
		elif args[1] == 'arn':
			print(arn)
		elif args[1] == 'info':
			quoted_tag = "'" + ctx.tag + "'"
			print("Getting instance info for " + quoted_tag)
			response = ec2_cli.describe_instances(Filters=[{ 'Name': 'tag:Type', 'Values': [eval(quoted_tag)] }])

			running = 0
			for res in response['Reservations']:
				for inst in res['Instances']:
					if inst['State']['Name'] == 'running':
						running += 1

			print("User:", ctx.user)
			print("Key:", ctx.key)
			print("Tag:", ctx.tag)
			print("Number of Instances:", len(ctx.instances))
			print(str(running) + " out of " + str(len(ctx.instances)) + " instances running")
		elif args[1] == 'help':
			help_str = ("Usage: python3 -m ec2_manager info\n"
						"	    python3 -m ec2_manager all <command>\n"
						"	    python3 -m ec2_manager <Node-ID> <command>\n\n"
						"Commands\n"
						"\tid:\tGet the instance-id of the node\n"
						"\tprip:\tGet the private ip address of the node\n"
						"\tpubip:\tGet the public ip address\n"
						"\tssh:\tConnect to a node\n"
						"\tsend:\tUse scp to send a file to the home directory\n"
						"\tget:\tUse scp to get a specific file from the node\n"
						"\tstart:\tStart the specified node or nodes\n"
						"\tstop:\tStop the specified node or nodes\n"
						"\treboot:\tRestart the current node\n"
						"\tcheck:\tCheck the current state of the node\n")
			print(help_str)
		else:
			print("Unrecognized option: use 'python3 -m ec2_manager help' to see options")

	return ctx


def main(args):
	ctx = Context()
	# check to see if resource file already exists. If not creates it
	res_exists = os.path.isfile(RES_DIR + 'context.b')
	if not res_exists:
		if args[1] == 'setup':
			if len(args) < 5:
				print("For setup, need '--user' '--key' and '--tag'")
				return
			else:
				if not os.path.isfile(RES_DIR):
					os.mkdir(RES_DIR)

				opts = parse_setup_opts(args[2:])
				instances = get_instances_info(opts.tag)
				ctx = Context(opts.user, opts.key, opts.tag, instances)
				pickle.dump(ctx, open(RES_DIR + 'context.b', 'wb'))
				return
		else:
			print("The res file does not exist... Have you run 'setup' yet?")
			return
	else:
		ctx = pickle.load( open(RES_DIR + 'context.b', 'rb') )

	ctx = process_options(ctx, args)

	pickle.dump(ctx, open(RES_DIR + 'context.b', 'wb'))
