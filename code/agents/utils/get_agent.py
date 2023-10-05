"""
Helper functions
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging


def get_agent(config):
    """
    Return a agent

    Args:
        config (config node object)
    Returns:
        agent (object)
    """
    if config.agent.agent_name == 'base':
        from agents.base_agent import BaseAgent
        agent = BaseAgent(config)

    elif config.agent.agent_name == 'exp_agent':
        from agents.exp_agent import ExpAgent
        agent = ExpAgent(config)
        
    elif config.agent.agent_name == 'exp_agent_org':
        from agents.exp_agent_org import ExpAgent
        agent = ExpAgent(config)

    elif config.agent.agent_name=='exp_agent_org_':
        from agents.exp_agent_org_ import ExpAgent
        agent = ExpAgent(config)
    elif config.agent.agent_name=='exp_agent_org_tta':
        from agents.exp_agent_org_tta import ExpAgent
        agent = ExpAgent(config)
    elif config.agent.agent_name=='exp_agent_org_valid_test':
        from agents.exp_agent_org_valid_test import ExpAgent
        agent = ExpAgent(config)
    else:
        logging.getLogger('Get Agent').error('Agent %s not implemented',
                                             config.agent.agent_name)
        raise NotImplementedError

    return agent
