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
    if config.model.model_name == 'unet_without_concat':#导入对应模型
        from models.unet_without_concat.unet_without_concat import UNet#不存在
        model = UNet(config)
    elif config.model.model_name == 'unet':
        from models.unet.unet import UNet
        model = UNet(config)
    elif config.model.model_name == 'unet_dep':
        from models.unet_dep.unet import UNet
        model = UNet(config)
    elif config.model.model_name == 'unet_baseline':
        from models.unet_baseline.unet_baseline import UNet
        model = UNet(config)
    elif config.model.model_name == 'unet_HAblock':
        from models.unet_HAblock.unet_HAblock import UNet
        model = UNet(config)
    elif config.model.model_name == 'unet_resblock':
        from models.unet_resblock.unet_resblock import UNet
        model = UNet(config)
    elif config.model.model_name == 'oct_unet':
        from models.octave.oct_unet import OctUNet
        model = OctUNet()
    elif config.model.model_name == 'dff':
        from models.dff.dff import DFF
        model = DFF()
    
    elif config.model.model_name == 'x1_net':
        from models.xnet.x1_net import x1_net
        model = x1_net()
    elif config.model.model_name == 'unet_cbam':
        from models.unet_CBAM.unet_CBAM import UNet
        model = UNet(config)
    elif config.model.model_name == 'unet_se':
        from models.unet_SE.unet_SE import UNet
        model = UNet(config)
    elif config.model.model_name == 'unet_DA':
        from models.unet_DAnet.unet_DAnet import UNet
        model = UNet(config)
    elif config.model.model_name == 'unet_ECA':
        from models.unet_ECA.unet_ECA import UNet
        model = UNet(config)
    elif config.model.model_name == 'unet_Chen':
        from models.unet_Chen.unet_Chen import UNet
        model = UNet(config)
    elif config.model.model_name == 'attention_unet':
        from models.attention_unet.attention_unet import AttU_Net
        model = AttU_Net()
    elif config.model.model_name == 'dmanet':
        from models.dmanet.dmanet import DMANet
        model = DMANet()

    else:
        logging.getLogger('Get Model').error(
            'Model for %s not implemented', config.model.model_name)
        raise NotImplementedError

    return model
