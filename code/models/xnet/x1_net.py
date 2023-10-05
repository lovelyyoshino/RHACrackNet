#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:19:58 2019

@author: BenzhangQiu
"""


import torch.nn as nn
from .module.XNet_module import module_init,Block_1,Block_2,module_end,ASPP, Decoder,module_fuse
import torch
import torch.nn.functional as F
    
class x1_net(nn.Module):
    def __init__(self,in_ch_1=3,out_ch=1):
        super(x1_net, self).__init__()
        
        self.init = module_init(in_ch_1,64,stride=2)
        
        self.block_1 = Block_1(64, 128, dilation=1, stride = 2)        
        self.block_2 = Block_1(128, 256, dilation=1, stride = 2)        
        self.block_3 = Block_1(256, 512, dilation=1, stride = 2)
        self.block_4 = Block_1(512, 1024, dilation=1, stride = 2)
#        self.block_RGB_3 = Block_1(256, 728, dilation=1, stride = 2, start_relu = True)
#        self.block_D_3 = Block_1(256, 728, dilation=1, stride = 2, start_relu = True)
        
#        self.block_20 = Block_1(728, 1024, dilation=1, stride = 1, start_relu = True)
        self.module_fuse = module_fuse(64,128,256,512,1024,32)
        self.end = module_end(160,out_ch)
        #self.ASPP = ASPP(256,256)
#        self.decode = Decoder(64,out_ch)
        
        
    def forward(self,input_):
        init = self.init(input_)
        
        x_1 = self.block_1(init)                
        x_2 = self.block_2(x_1)  
        x_3 = self.block_3(x_2)  
        x_4 = self.block_4(x_3)  

        out = self.module_fuse(init,x_1,x_2,x_3,x_4)
        out = self.end(out)
        return out

if __name__ == '__main__':
    model = x1_net(3,1)
    model.eval()
    image_1 = torch.randn(1, 3, 480, 640)

    with torch.no_grad():
        output = model.forward(image_1)
    print(output.size())