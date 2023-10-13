"""
Main script
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from util.get_config import get_config
from agents.utils.get_agent import get_agent


def main():
    """
    Main process of this project
    """
    config = get_config()   
    agent = get_agent(config)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()

