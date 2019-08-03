import os

import ec2_manager


class Instance:
    """
    Class for an AWS EC2 node.
    """

    def __init__(self, _id, _pr_ip, _pub_ip):
        self.id = _id
        self.pr_ip = _pr_ip
        self.pub_ip = _pub_ip


class Context:
    """
    Class for an EC2 manager context, managing over a group (cluster) of nodes.
    """
    
    def __init__(self, _name = '', _user = '', _key = '', _tag = '', _instances = []):
        self.name = _name
        self.user = _user
        self.key = _key
        self.tag = _tag
        self.instances = _instances
