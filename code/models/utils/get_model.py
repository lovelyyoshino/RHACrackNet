"""
Helper functions
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging


def get_model(config):
    """
    Return a model object

    Args:
        config (config node object): config

    Returns:
        model (torch.nn.Module object): pytorch model
    """
    if config.model.model_name == 'x1_net':#导入对应模型
        from models.xnet.x1_net import x1_net
        model = x1_net()
    elif config.model.model_name == 'RHANet':
        from models.RHANet.rhanet import RHANet
        model = RHANet(config)
    elif config.model.model_name == 'RHANet_dep':
        from models.RHANet_dep.rhanet_dep import RHANet
        model = RHANet(config)
    elif config.model.model_name == 'unet_cbam':
        from models.unet_CBAM.unet_CBAM import UNet
        model = UNet(config)
    elif config.model.model_name == 'attention_unet':
        from models.attention_unet.attention_unet import AttU_Net
        model = AttU_Net()

    else:
        logging.getLogger('Get Model').error(
            'Model for %s not implemented', config.model.model_name)
        raise NotImplementedError

    return model
