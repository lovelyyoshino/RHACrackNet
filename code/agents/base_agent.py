"""
The Base Agent class contains definitions for all the necessary functions,
     other agents should inherit from the base.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class BaseAgent(object):
    """
    Base agent class

    Args:
        config (config node object): the given config for the agent
    """

    def __init__(self, config):
        self.config = config

    def load_ckpt(self, ckpt_name):
        """
        Load checkpoint with given ckpt_name

        Args:
            ckpt_name (string): the name of checkpoint to load

        """
        raise NotImplementedError

    def save_ckpt(self, ckpt_name='ckpt.pth', is_best=False):
        """
        Save the current state_dict of agent model to ckpt_path

        Args:
            ckpt_name (string, optional): the name of the current state_dict to
                 save as
            is_best (bool, optional): indicator for whether the model is best
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator of agent.
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """
        One epoch of training
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes all the operations of the operator and the data loader
        """
        raise NotImplementedError
