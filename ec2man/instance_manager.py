import os
import boto3
import argparse

## Allocate new instances
def launch_ec2_instances(input_args):
	ec2 = boto3.resource('ec2')
	instances = []

	parser = argparse.ArgumentParser()
	parser.add_argument('--ami', type=str)
	parser.add_argument('--type', type=str)
	parser.add_argument('--cnt', type=int, default=1)
	parser.add_argument('--ctx', type=str, default=None)
	parser.add_argument('--az', type=str, default=None)
	parser.add_argument('--sg', type=str, default=None)

	opts = parser.parse_args(input_args)

	if opts.ctx== None:
		print("Must provide a context for the allocated instances")
		return 13

	args = {
		'ImageId': opts.ami,
		'InstanceType': opts.type,
		'MinCount': opts.cnt,
		'MaxCount': opts.cnt,
		'SecurityGroupIds': ['sg-0a98f6952f8c78610']
	}

	if opts.type != 't2.micro':
		args['EbsOptimized'] = True

	if opts.az != None:
		args['Placement'] = { 'AvailabilityZone': opts.az }
		if opts.az[:-1] == 'us-east-2':
			args['SecurityGroupIds'] = ['sg-098524cf5a5d0011f'],

	print("Instance options:", args)
	response = ec2.create_instances(**args)
	instance_ids = [inst.id for inst in response]

	print("Adding " + str(len(instance_ids)) + " machines to context " + opts.ctx)
	with open('ec2man/machines', 'a') as f:
		f.write('\n')
		for ec2_id in instance_ids:
			f.write(" ".join([ec2_id, opts.ctx]) + '\n')

def add_managed_ec2_instance(input_args):
	context = input_args[0]
	instances = input_args[1:]
	print("Adding " + str(len(instances)) + " machines to context " + context)
	with open('ec2man/machines', 'a') as f:
		f.write('\n')
		for ec2_id in instances:
			f.write(" ".join([ec2_id, context]) + '\n')
