import os

import ec2_manager


class Instance:
    def __init__(self, _id, _pr_ip, _pub_ip):
        self.id = _id
        self.pr_ip = _pr_ip
        self.pub_ip = _pub_ip
        

class Context:
	def __init__(self, _user = '', _key = '', _tag = '',  _instances = []):
		self.user = _user
		self.key = _key
		self.tag = _tag
		self.instances = _instances
