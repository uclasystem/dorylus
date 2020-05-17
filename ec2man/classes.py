import os

import ec2man


class Instance:
    """
    Class for an AWS EC2 node.
    """

    def __init__(self, _id='', _type='', _placement = '', _pr_ip='', _pub_ip='0', _user='', _key=''):
        self.id = _id
        self.type = _type
        self.placement = _placement
        self.pr_ip = _pr_ip
        self.pub_ip = _pub_ip
        self.user = _user
        self.key = _key

    def set_user_key(self, user_key_tuple):
        """
        Set the user and key field.
        """

        self.user, self.key = user_key_tuple


class Context:
    """
    Class for an EC2 manager context, managing over a group (cluster) of nodes.
    """
    
    def __init__(self, _name='', _instances=[]):
        self.name = _name
        self.instances = _instances
