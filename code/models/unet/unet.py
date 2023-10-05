"""
UNet models
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
from torch.nn import functional as F
import torch 

from .modules.unet_modules_depthwise import InitConv,EncoderBlock,DecoderBlock,FinalConv,ResNet_Each_Conv,ResBlock,FinalLoss_Crack,ResBlock_Norml,DecoderBlock_Conv1,Attention_block,ResNet_First_Conv,mid_output_get,Augmented_Attention_block
#from models.unet.modules.unet_modules import EncoderBlock
#from models.unet.modules.unet_modules import DecoderBlock
#from models.unet.modules.unet_modules import FinalConv
from torchsummary import summary

class UNet(nn.Module):
    # def __init__(self, config):
    def __init__(self):
        super(UNet, self).__init__()#继承自父类的属性进行初始化
        # self.config=config
        input_ch = 3
        init_conv_mid_ch, init_conv_out_ch = 16, 16#对应的中间和输出参数
        encode_1_mid_ch, encode_1_out_ch = 16, 32
        encode_2_mid_ch, encode_2_out_ch = 32, 64
        encode_3_mid_ch, encode_3_out_ch = 64, 128
        encode_4_mid_ch, encode_4_out_ch = 128, 128
        decode_1_mid_ch, decode_1_out_ch = 128, 64
        decode_2_mid_ch, decode_2_out_ch = 64, 32
        decode_3_mid_ch, decode_3_out_ch = 32, 16
        decode_4_mid_ch, decode_4_out_ch = 16, 16
        output_ch = 1

        self.init_conv = InitConv(inC=input_ch, midC=init_conv_mid_ch,
                                  outC=init_conv_out_ch, use_bias=True,
                                  midK=3, midS=1, midP=1, midD=1, midG=1,
                                  outK=3, outS=1, outP=1, outD=1, outG=1)

        self.encode_1 = EncoderBlock(scale_factor=2, pool_type='max',
                                     inC=init_conv_out_ch, midC=encode_1_mid_ch,
                                     outC=encode_1_out_ch, use_bias=True,
                                     midK=3, midS=1, midP=1, midD=1, midG=1,
                                     outK=3, outS=1, outP=1, outD=1, outG=1)

        self.encode_2 = EncoderBlock(scale_factor=2, pool_type='max',
                                     inC=encode_1_out_ch, midC=encode_2_mid_ch,
                                     outC=encode_2_out_ch, use_bias=True,
                                     midK=3, midS=1, midP=1, midD=1, midG=1,
                                     outK=3, outS=1, outP=1, outD=1, outG=1)

        self.encode_3 = EncoderBlock(scale_factor=2, pool_type='max',
                                     inC=encode_2_out_ch, midC=encode_3_mid_ch,
                                     outC=encode_3_out_ch, use_bias=True,
                                     midK=3, midS=1, midP=1, midD=1, midG=1,
                                     outK=3, outS=1, outP=1, outD=1, outG=1)

        self.encode_4 = EncoderBlock(scale_factor=2, pool_type='max',
                                     inC=encode_3_out_ch, midC=encode_4_mid_ch,
                                     outC=encode_4_out_ch, use_bias=True,
                                     midK=3, midS=1, midP=1, midD=1, midG=1,
                                     outK=3, outS=1, outP=1, outD=1, outG=1)

        self.resencode_4 = ResNet_Each_Conv(block=ResBlock, input_channel=encode_4_mid_ch,
                                            output_channel=encode_4_out_ch, block_num=1, stride=1)
        self.resencode_4_1 = ResNet_Each_Conv(block=ResBlock, input_channel=encode_4_mid_ch,
                                            output_channel=encode_4_out_ch, block_num=1, stride=1)

        self.decode_1 = DecoderBlock_Conv1(inC=encode_4_out_ch, trpC=encode_4_out_ch,
                                     midC=decode_1_mid_ch, outC=decode_1_out_ch,
                                     use_bias=True, scale_factor=2,
                                     decoder_type='transposed',
                                     trpK=3, trpP=1, trpOP=1, trpD=1, trpG=1,
                                     midK=3, midS=1, midP=1, midD=1, midG=1,
                                     outK=3, outS=1, outP=1, outD=1, outG=1)

        self.decode_2 = DecoderBlock_Conv1(inC=decode_1_out_ch, trpC=decode_1_out_ch,
                                     midC=decode_2_mid_ch, outC=decode_2_out_ch,
                                     use_bias=True, scale_factor=2,
                                     decoder_type='transposed',
                                     trpK=3, trpP=1, trpOP=1, trpD=1, trpG=1,
                                     midK=3, midS=1, midP=1, midD=1, midG=1,
                                     outK=3, outS=1, outP=1, outD=1, outG=1)

        self.attention_GAP=Augmented_Attention_block(encode_4_mid_ch,encode_4_out_ch,decode_1_mid_ch)

        self.decode_3 = DecoderBlock_Conv1(inC=decode_2_out_ch, trpC=decode_2_out_ch,
                                     midC=decode_3_mid_ch, outC=decode_3_out_ch,
                                     use_bias=True, scale_factor=2,
                                     decoder_type='transposed',
                                     trpK=3, trpP=1, trpOP=1, trpD=1, trpG=1,
                                     midK=3, midS=1, midP=1, midD=1, midG=1,
                                     outK=3, outS=1, outP=1, outD=1, outG=1)

        self.decode_4 = DecoderBlock_Conv1(inC=decode_3_out_ch, trpC=decode_3_out_ch,
                                     midC=decode_4_mid_ch, outC=decode_4_out_ch,
                                     use_bias=True, scale_factor=2,
                                     decoder_type='transposed',
                                     trpK=3, trpP=1, trpOP=1, trpD=1, trpG=1,
                                     midK=3, midS=1, midP=1, midD=1, midG=1,
                                     outK=3, outS=1, outP=1, outD=1, outG=1)

        self.final_conv = FinalConv(inC=decode_4_out_ch, outC=output_ch,
                                    activate_fn='sigmoid', use_bias=True,
                                    outK=3, outS=1, outP=1, outD=1, outG=1)
        self.final_deep = FinalLoss_Crack()
        self.pool=nn.AvgPool2d(kernel_size=2,
                                 stride=None, padding=0,
                                 ceil_mode=False,
                                 count_include_pad=True)

        self.size_x = 0
        self.size_y = 0

    def forward(self, input_):
        print(type(input_))
        # input_x=input_.cpu().detach().numpy().shape[2];input_y=input_.cpu().detach().numpy().shape[3]
        # print(input_.cpu().detach().numpy().shape[2],">>>>",input_.cpu().detach().numpy().shape[3])

        input_ = self._same_padding(input_)
        init_conv = self.init_conv(input_)

        encode_1 = self.encode_1(init_conv)
        encode_2 = self.encode_2(encode_1)
        encode_3 = self.encode_3(encode_2)
        encode_4 = self.encode_4(encode_3)

        encode_4_mid=encode_4
        encode_4 = self.resencode_4(encode_4)
        encode_4 = self.resencode_4_1(encode_4)
        encode_4_end = self.attention_GAP(encode_4_mid,encode_4)

        decode_1 = self.decode_1(encode_4_end, encode_3)
        decode_2 = self.decode_2(decode_1,encode_2)
        decode_3 = self.decode_3(decode_2, encode_1)
        decode_4 = self.decode_4(decode_3, init_conv)
        output = self.final_conv(decode_4)

        if  0:
            output=self.pool(output)
        output = output[:,:,0:self.size_x,0:self.size_y]
        return output

    def _same_padding(self,input_):
        self.num = 16
        self.size_x = input_.size(2)
        self.size_y = input_.size(3)
        x_padding_num = 0
        y_padding_num = 0
        if self.size_x % self.num != 0:
            x_padding_num = (self.size_x//self.num+1)*self.num - self.size_x
        if self.size_y % self.num != 0:
            y_padding_num = (self.size_y//self.num+1)*self.num - self.size_y

        pad_parten = (0, y_padding_num, 0, x_padding_num)
        output = F.pad(input=input_, pad=pad_parten,
                               mode='constant', value=0)
        return output

if __name__ == '__main__':
    unet = UNet()
    unet=unet.to("cuda:0")
    summary(unet,(3,572,572))
    #a = torch.randn((1,3,311,462))
   # print(a.size())
    #b = unet(a)
