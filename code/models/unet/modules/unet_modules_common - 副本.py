"""
Modules of basic convolutional layers
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from torch.nn import functional as F
from .dyrelu import DyReLUB

def mid_output_get():
    global  output_mid
    return output_mid

# class Depthwise_Conv(nn.Module):
#     def __init__(self, input_channel, output_channel,kernel_size,stride):
#         super(Depthwise_Conv, self).__init__()
#         self.depth_conv = nn.Conv2d(
#             in_channels=input_channel,
#             out_channels=input_channel,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=1,
#             groups=input_channel
#         )
#         self.point_conv = nn.Conv2d(
#             in_channels=input_channel,
#             out_channels=output_channel,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1
#         )
#
#     def forward(self, input_):
#         output = self.depth_conv(input_)
#         output = self.point_conv(output)
#         return output

class ResBlock_Norml(nn.Module):
    def __init__(self,input_channel,output_channel,stride=1):#输入输出以及stride采样间隔
        super(ResBlock_Norml, self).__init__()
        # 残差块的第一个卷积
        # 通道数变换in->out，每一层（除第一层外）的第一个block
        #self.conv_1=Depthwise_Conv(input_channel,output_channel,kernel_size=1,stride=1)
        self.conv_1=nn.Conv2d(input_channel,output_channel,kernel_size=3, stride=stride, padding=1)
        self.bn_1 = nn.BatchNorm2d(output_channel)
        self.relu = DyReLUB(output_channel,conv_type='2d')#nn.ReLU(inplace=True)
        # 残差块的第二个卷积
        # 通道数、图片尺寸均不变
        # self.conv_2 = Depthwise_Conv(input_channel,output_channel,kernel_size=1,stride=1)
        self.conv_2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(output_channel)
        # 残差块的shortcut
        # 如果残差块的输入输出通道数不同，则需要变换通道数及图片尺寸，以和residual部分相加
        # 输出：通道数*2 图片尺寸/2
        if True:#input_channel != output_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=2),
                nn.BatchNorm2d(output_channel)
            )
        else:
            # 通道数相同，无需做变换，在forward中identity = x
            self.downsample = None


    def forward(self,input_):
        global output_mid
        identity=input_
        output=self.conv_1(input_)
        output=self.bn_1(output)
        output=self.relu(output)
        output_mid=output
        output=self.conv_2(output)
        output=self.bn_2(output)
        if self.downsample is not None:
            identity = self.downsample(identity)
        #print(output.shape,identity.shape)
        output += identity
        output = self.relu(output)

        return output

class ResBlock(nn.Module):
    def __init__(self,input_channel,output_channel,stride=1):#输入输出以及stride采样间隔
        super(ResBlock, self).__init__()
        # 残差块的第一个卷积
        # 通道数变换in->out，每一层（除第一层外）的第一个block
        # self.conv_1 = Depthwise_Conv(input_channel,output_channel,kernel_size=3,stride=1)#
        self.conv_1 = nn.Conv2d(input_channel,output_channel,kernel_size=3, stride=stride, padding=1)
        self.bn_1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        # 残差块的第二个卷积
        # 通道数、图片尺寸均不变
        # self.conv_2 = Depthwise_Conv(output_channel,output_channel,kernel_size=3,stride=1)#
        self.conv_2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(output_channel)
        # 残差块的shortcut
        # 如果残差块的输入输出通道数不同，则需要变换通道数及图片尺寸，以和residual部分相加
        # 输出：通道数*2 图片尺寸/2
        if input_channel != output_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=2),
                nn.BatchNorm2d(output_channel)
            )
        else:
            # 通道数相同，无需做变换，在forward中identity = x
            self.downsample = None


    def forward(self,input_):
        global output_mid
        identity=input_
        output=self.conv_1(input_)
        output=self.bn_1(output)
        output=self.relu(output)
        output_mid=output
        output=self.conv_2(output)
        output=self.bn_2(output)
        if self.downsample is not None:
            identity = self.downsample(identity)
        #print(output.shape,identity.shape)
        output += identity
        output = self.relu(output)

        return output

class ResNet_First_Conv(nn.Module):
    def __init__(self):
        super(ResNet_First_Conv, self).__init__()
            # 初始卷积层核池化层
        self.first = nn.Sequential(
            # 卷基层1：7*7kernel，2stride，3padding，outmap：32-7+2*3 / 2 + 1，16*16
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            DyReLUB(64,conv_type='2d'),#nn.ReLU(inplace=True),
            # 最大池化，3*3kernel，1stride（32的原始输入图片较小，不再缩小尺寸），1padding，
            # outmap：16-3+2*1 / 1 + 1，16*16
            nn.MaxPool2d(3, 1, 1)
        )
    def forward(self,input_):
        output=self.first(input_)
        return output

class ResNet_Each_Conv(nn.Module):
    def __init__(self, block,input_channel,output_channel,block_num=2,stride=2):
        super(ResNet_Each_Conv, self).__init__()
        self.block=block
        self.input_channel=input_channel
        self.output_channel=output_channel
        self.block_num=block_num
        self.stride=stride
        self.expansion=1#输出为1倍
        self.layers=self.make_layer(self.block,self.input_channel,self.output_channel,self.block_num,self.stride)

    def make_layer(self, block, inplane,outplanes, blocks, stride=2):
        layers=[]
        # 每一层的第一个block，通道数可能不同
        layers.append(block(inplane, outplanes, stride))
        inplanes = outplanes * self.expansion  # 记录layerN的channel变化，具体请看ppt resnet表格
        # 每一层的其他block，通道数不变，图片尺寸不变
        for i in range(self.block_num - 1):
            layers.append(block(inplanes, outplanes, 1))
        return nn.Sequential(*layers)

    def forward(self,input_):
        output=self.layers(input_)
        return output

class FinalLoss_Crack(nn.Module):
    def __init__(self):
        super(FinalLoss_Crack, self).__init__()
        # self.conv_1 = Depthwise_Conv(256,64,kernel_size=1,stride=1)
        self.conv_1 = nn.Conv2d(256, 64, kernel_size=1, stride=1)#卷积
        # self.conv_2 = Depthwise_Conv(128,64,kernel_size=1,stride=1)
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=1, stride=1)#卷积
        #self.conv_3=nn.Conv2d(64, 64, kernel_size=1, stride=1)#卷积
        #self.conv_4=nn.Conv2d(64, 64, kernel_size=1, stride=1)#卷积
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.conv_4=nn.ConvTranspose2d(64, 64, kernel_size, stride=1, padding=0, output_padding=0, bias=True)
        # self.fuse_conv = Depthwise_Conv(64*2,64,kernel_size=1,stride=1)#
        self.fuse_conv = nn.Conv2d(64*2, 64, kernel_size=1, stride=1)
        # self.conv_f =Depthwise_Conv(64,1,kernel_size=1,stride=1)#
        self.conv_f = nn.Conv2d(in_channels=64, out_channels=1,kernel_size=1, stride=1, bias=True)#输出一个参数
        self.act = DyReLUB(64,conv_type='2d')#nn.ReLU(inplace=True)
    def forward(self,loss1,loss2,loss3,loss4,W,H):
        self.h=H
        self.w=W
        #loss1=self.conv_1(loss1)
        #loss1=self.bn(loss1)
        #loss2=self.conv_2(loss2)
        #loss2 = self.bn(loss2)
        #side_output1 = F.interpolate(loss1, size=(self.h, self.w), mode='bilinear', align_corners=True)
        #side_output2 = F.interpolate(loss2, size=(self.h, self.w), mode='bilinear', align_corners=True)
        side_output3 = F.interpolate(loss3, size=(self.h, self.w), mode='bilinear', align_corners=True)
        side_output3=self.bn(side_output3)
        side_output3=self.relu(side_output3)
        #print(side_output1.shape, ">>>", side_output2.shape, ">>>", side_output3.shape, ">>>", loss4.shape)
        side_output4 = F.interpolate(loss4, size=(self.h, self.w), mode='bilinear', align_corners=True)
        side_output4=self.bn(side_output4)
        side_output4=self.relu(side_output4)
        #side_output4 = F.interpolate(self.loss4, size=(h, w), mode='bilinear', align_corners=True)
        #print(side_output1.shape,">>>",side_output2.shape,">>>",side_output3.shape,">>>",side_output4.shape)
        fused = self.fuse_conv(torch.cat([side_output3,
                                        side_output4], dim=1))
        # fused = torch.add(side_output3*0.3,
        #                   side_output4*0.7)
        output=self.act(self.conv_f(fused))
        #print(output.shape)
        return output#side_output1,side_output2,side_output3,side_output4,fused

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            # Depthwise_Conv(F_g,F_int,kernel_size=1,stride=1),
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),#卷积F_g到F_int
            nn.BatchNorm2d(F_int)#归一化F_int
            )

        self.W_x = nn.Sequential(
            # Depthwise_Conv(F_l, F_int, kernel_size=1, stride=1),
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            # Depthwise_Conv(F_int, 1, kernel_size=1, stride=1),
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        #print(">>>>>XXXX",g.shape,x.shape)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        #print(">>>>>",g1.shape,x1.shape)
        psi = self.relu(g1+x1)#两个累加
        psi = self.psi(psi)#最终形成1个channel的维度比重
        #output = x*psi
        #print(x.shape,">>>>",psi.shape,"""""",(x*psi).shape)
        return x*psi


class Augmented_Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Augmented_Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            #Depthwise_Conv(F_g,F_int,kernel_size=1,stride=1),
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),#卷积F_g到F_int
            nn.BatchNorm2d(F_int)#归一化F_int
            )
        self.W_gapg = nn.Sequential(
            nn.AdaptiveAvgPool2d((2,2)),  # 自适应池化，指定池化输出尺寸为 1 * 1
            #Depthwise_Conv(F_l, F_int, kernel_size=1, stride=1),
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            #Depthwise_Conv(F_l, F_int, kernel_size=1, stride=1),
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_gapx = nn.Sequential(
            nn.AdaptiveAvgPool2d((2,2)),  # 自适应池化，指定池化输出尺寸为 1 * 1
            #Depthwise_Conv(F_l, F_int, kernel_size=1, stride=1),
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            #Depthwise_Conv(F_int, 1, kernel_size=1, stride=1),
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(kernel_size=2,
                                 stride=None, padding=0,
                                 ceil_mode=False,
                                 count_include_pad=True)

        self.conv_m=nn.Conv2d(F_int, F_int, kernel_size=1,stride=1,padding=0,bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax2d()

    def forward(self, g, x):
        #print(g.shape)
        g1_gap = self.W_gapg(g)
        x1_gap = self.W_gapx(x)

        mutil=self.W_gapx(g+x)
        mutil = self.pool(mutil)

        mutil_x = x1_gap * mutil
        mutil_x_g = self.relu( mutil_x + g1_gap )

        mutil_x_g=self.conv_m(mutil_x_g)
        mutil_x_g=self.softmax(mutil_x_g)
        mutil_x_g=self.pool(mutil_x_g)

        M_channel = mutil_x_g * x

        g_s = self.W_g(g)
        F_s = self.W_x(M_channel)

        M_spatial = self.relu( F_s + g_s )
        psi = self.psi( M_spatial )
        output = x * psi
        # output = g * psi
        return output

class SingleReLUBNConv(nn.Module):
    """
    A block of 2 ReLU activated batch normalized convolution layer
    """

    def __init__(self, inC, midC, outC, use_bias=True,
                 midK=3, midS=1, midP=1, midD=1, midG=1,
                 outK=3, outS=1, outP=1, outD=1, outG=1):
        super(SingleReLUBNConv, self).__init__()

        self.inC, self.midC, self.outC = inC, midC, outC
        self.use_bias = use_bias

        self.midK, self.midS, self.midP, self.midD = midK, midS, midP, midD
        self.midG = midG

        self.outK, self.outS, self.outP, self.outD = outK, outS, outP, outD
        self.outG = outG
        # self.conv_1= self.conv_m=Depthwise_Conv(inC, outC, kernel_size=1, stride=1)
        self.conv_1= self.conv_m = nn.Conv2d(inC, outC, kernel_size=3, stride=midS, padding=1)
        self.bn_1 = nn.BatchNorm2d(num_features=outC, eps=1e-5, momentum=0.1,#卷积层之后总会添加BatchNorm2d进行数据的归一化处理
                                   affine=True, track_running_stats=True)

        self.relu = nn.ReLU(inplace=True)#激活函数

    def forward(self, input_):
        output = self.relu(self.bn_1(self.conv_1(input_)))#输出中间572*572*1-->570*570*64
        return output

class DoubleReLUBNConv(nn.Module):
    """
    A block of 2 ReLU activated batch normalized convolution layer
    """

    def __init__(self, inC, midC, outC, use_bias=True,
                 midK=3, midS=1, midP=1, midD=1, midG=1,
                 outK=3, outS=1, outP=1, outD=1, outG=1):
        super(DoubleReLUBNConv, self).__init__()

        self.inC, self.midC, self.outC = inC, midC, outC
        self.use_bias = use_bias

        self.midK, self.midS, self.midP, self.midD = midK, midS, midP, midD
        self.midG = midG

        self.outK, self.outS, self.outP, self.outD = outK, outS, outP, outD
        self.outG = outG
        self.conv_1=nn.Conv2d(inC, midC, kernel_size=3, stride=midS, padding=1)
        # self.conv_1 = nn.Conv2d(in_channels=inC, out_channels=midC,
        #                         kernel_size=midK, stride=midS,
        #                         padding=midP, dilation=midD, groups=midG,
        #                         bias=use_bias)
        #in_channels输入数据的通道数；out_channels输出数据的通道数；midK卷积核；midS步长（过几个采样一次），默认为1；midP对原来的输入层基础上，上下左右各补了一行；midD为kernel间距，每个核函数间距1也就是3*3大小形成5*5；
        #groups将输入的feature map的channel分成n组，分组卷积；bias如果为真，则将可学习的偏差添加到输出中。默认值：True
        self.bn_1 = nn.BatchNorm2d(num_features=midC, eps=1e-5, momentum=0.1,#卷积层之后总会添加BatchNorm2d进行数据的归一化处理
                                   affine=True, track_running_stats=True)#1.num_features：一般输入参数为batch_size*num_features*height*width，即为其中特征的数量
                                                                         #2.eps：分母中添加的一个值，目的是为了计算的稳定性，默认为：1e-5
                                                                         #3.momentum：一个用于运行过程中均值和方差的一个估计参数（我的理解是一个稳定系数，类似于SGD中的momentum的系数）
                                                                         #4.affine：当设为true时，会给定可以学习的系数矩阵gamma和beta
        self.conv_2=nn.Conv2d(midC, outC, kernel_size=3, stride=outS, padding=1)
        # self.conv_2 = nn.Conv2d(in_channels=midC, out_channels=outC,
        #                         kernel_size=outK, stride=outS,
        #                         padding=outP, dilation=outD, groups=outG,
        #                         bias=use_bias)
        self.bn_2 = nn.BatchNorm2d(num_features=outC, eps=1e-5, momentum=0.1,
                                   affine=True, track_running_stats=True)

        self.relu = nn.ReLU(inplace=True)#激活函数

    def forward(self, input_):
        global output_mid
        output = self.relu(self.bn_1(self.conv_1(input_)))#输出中间572*572*1-->570*570*64
        output_mid=output
        output = self.relu(self.bn_2(self.conv_2(output)))#输出一层结果570*570*64-->568*568*64
        return output


class InitConv(nn.Module):
    def __init__(self, **kwargs):#导入对应的字典
        super(InitConv, self).__init__()
        self.double_conv = DoubleReLUBNConv(**kwargs)#2个ReLU激活的批次归一化卷积层的块

    def forward(self, input_):
        output = self.double_conv(input_)
        return output


class EncoderBlock(nn.Module):
    def __init__(self, scale_factor=2, pool_type='max', **kwargs):
        super(EncoderBlock, self).__init__()
        self.scale_factor = scale_factor
        self.pool_type = pool_type

        if self.pool_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size=scale_factor,
                                     stride=None, padding=0, dilation=1,
                                     return_indices=False, ceil_mode=False)#kernel_size(int or tuple) - max pooling的窗口大小，可以为tuple，在nlp中tuple用更多，（n,1）
                                                                           #stride(int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size,None为默认值
                                                                           #padding(int or tuple, optional) - 输入的每一条边补充0的层数
                                                                           #dilation(int or tuple, optional) – 一个控制窗口中元素步幅的参数
                                                                           #return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助
                                                                           #ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作


        elif self.pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=scale_factor,
                                     stride=None, padding=0,
                                     ceil_mode=False,
                                     count_include_pad=True)

        else:
            raise NotImplementedError

        self.double_conv = DoubleReLUBNConv(**kwargs)

    def forward(self, input_):
        output = self.double_conv(self.pool(input_))
        return output

class DecoderBlock_Conv1(nn.Module):
    def __init__(self, inC, trpC, midC, outC, use_bias=True,
                 scale_factor=2, decoder_type='transposed',
                 trpK=3, trpP=1, trpOP=1, trpD=1, trpG=1,
                 **kwargs):
        super(DecoderBlock_Conv1, self).__init__()

        self.use_bias = use_bias
        self.scale_factor = scale_factor
        self.decoder_type = decoder_type

        if decoder_type == 'transposed':
            self.up_scale = nn.ConvTranspose2d(in_channels=inC,
                                               out_channels=trpC,
                                               kernel_size=trpK,
                                               stride=scale_factor,
                                               padding=trpP,
                                               output_padding=trpOP,
                                               groups=trpG,
                                               bias=use_bias,
                                               dilation=trpD,
                                               padding_mode='zeros')#in_channels(int) – 输入信号的通道数
                                                                    #out_channels(int) – 卷积产生的通道数
                                                                    #kerner_size(int or tuple) - 卷积核的大小
                                                                    #stride(int or tuple,optional) - 卷积步长
                                                                    #padding(int or tuple, optional) - 输入的每一条边补充0的层数
                                                                    #output_padding(int or tuple, optional) - 输出的每一条边补充0的层数
                                                                    #dilation(int or tuple, optional) – 卷积核元素之间的间距
                                                                    #groups(int, optional) – 从输入通道到输出通道的阻塞连接数
                                                                    #bias(bool, optional) - 如果bias=True，添加偏置

        elif decoder_type == 'bilinear':
            self.up_scale = nn.Upsample(scale_factor=scale_factor,
                                        mode='bilinear',
                                        align_corners=None)#size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional) – 根据不同的输入类型制定的输出大小
                                                           #scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional) – 指定输出为输入的多少倍数。如果输入为tuple，其也要制定为tuple类型
                                                           #mode (str, optional) – 可使用的上采样算法，有'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'. 默认使用'nearest'
                                                           #align_corners (bool, optional) – 如果为True，输入的角像素将与输出张量对齐，因此将保存下来这些像素的值。仅当使用的算法为'linear', 'bilinear'or 'trilinear'时可以使用。默认设置为False

        else:
            raise NotImplementedError

        #self.double_conv = DoubleReLUBNConv(inC=trpC*2, midC=midC, outC=outC,
        #                                    use_bias=use_bias, **kwargs)
        # self.conv_1 = Depthwise_Conv(trpC*2, outC,kernel_size=3, stride=1)
        self.conv_1 = nn.Conv2d(trpC*2, outC, kernel_size=3, stride=1, padding=1)#Depthwise_Conv(trpC*2, outC, stride=1)
        self.bn_1 = nn.BatchNorm2d(num_features=outC, eps=1e-5, momentum=0.1,  # 卷积层之后总会添加BatchNorm2d进行数据的归一化处理
                                   affine=True,
                                   track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)  # 激活函数

        self.attention=Augmented_Attention_block(F_g=trpC,F_l=trpC,F_int=trpC)


    def forward(self, input_, skip):#链接
        output = self.up_scale(input_)

        skip=self.attention(output,skip)
        #output = self._same_padding(output, skip)
        #skip = self._same_padding(skip, output)
        # output = torch.cat([skip, output], dim=1)
        #print(skip.shape)
        #output = self.double_conv(output)

        output = torch.cat([skip, output], dim=1)

        output = self.relu(self.bn_1(self.conv_1(output)))  # 输出中间572*572*1-->570*570*64

        return output, skip

    @staticmethod
    def _same_padding(input_, target, data_format='NCHW'):#input_为same作为目标形状
        """
        Zero pad input_ as the shape of target
        """
        if data_format == 'NCHW':
            if input_.shape == target.shape:
                return input_

            else:
                assert input_.shape[:-2] == target.shape[:-2]
                _, h_input, w_input, _ = input_.shape
                _, h_target, w_target, _ = target.shape

                h_dff = h_target - h_input
                w_diff = w_target - w_input

                pad_parten = (h_dff // 2, h_dff - h_dff // 2,
                              w_diff // 2, w_diff - w_diff // 2)

                output = F.pad(input=input_, pad=pad_parten,
                               mode='constant', value=0)
                return output
        else:
            raise NotImplementedError

class DecoderBlock(nn.Module):
    def __init__(self, inC, trpC, midC, outC, use_bias=True,
                 scale_factor=2, decoder_type='transposed',
                 trpK=3, trpP=1, trpOP=1, trpD=1, trpG=1,
                 **kwargs):
        super(DecoderBlock, self).__init__()

        self.use_bias = use_bias
        self.scale_factor = scale_factor
        self.decoder_type = decoder_type

        if decoder_type == 'transposed':
            self.up_scale = nn.ConvTranspose2d(in_channels=inC,
                                               out_channels=trpC,
                                               kernel_size=trpK,
                                               stride=scale_factor,
                                               padding=trpP,
                                               output_padding=trpOP,
                                               groups=trpG,
                                               bias=use_bias,
                                               dilation=trpD,
                                               padding_mode='zeros')#in_channels(int) – 输入信号的通道数
                                                                    #out_channels(int) – 卷积产生的通道数
                                                                    #kerner_size(int or tuple) - 卷积核的大小
                                                                    #stride(int or tuple,optional) - 卷积步长
                                                                    #padding(int or tuple, optional) - 输入的每一条边补充0的层数
                                                                    #output_padding(int or tuple, optional) - 输出的每一条边补充0的层数
                                                                    #dilation(int or tuple, optional) – 卷积核元素之间的间距
                                                                    #groups(int, optional) – 从输入通道到输出通道的阻塞连接数
                                                                    #bias(bool, optional) - 如果bias=True，添加偏置

        elif decoder_type == 'bilinear':
            self.up_scale = nn.Upsample(scale_factor=scale_factor,
                                        mode='bilinear',
                                        align_corners=None)#size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional) – 根据不同的输入类型制定的输出大小
                                                           #scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional) – 指定输出为输入的多少倍数。如果输入为tuple，其也要制定为tuple类型
                                                           #mode (str, optional) – 可使用的上采样算法，有'nearest', 'linear', 'bilinear', 'bicubic' and 'trilinear'. 默认使用'nearest'
                                                           #align_corners (bool, optional) – 如果为True，输入的角像素将与输出张量对齐，因此将保存下来这些像素的值。仅当使用的算法为'linear', 'bilinear'or 'trilinear'时可以使用。默认设置为False

        else:
            raise NotImplementedError

        self.double_conv = SingleReLUBNConv(inC=trpC*2, midC=midC, outC=outC,
                                            use_bias=use_bias, **kwargs)

        self.attentions = Attention_block(F_g=trpC,F_l=midC,F_int=outC)

    def forward(self, input_, skip):#链接
        output = self.up_scale(input_)
        #print(output.shape,skip.shape)
        output_mid=self.attentions(output, x=skip)
        #output = self._same_padding(output, skip)
        #skip = self._same_padding(skip, output)
        #print(output_mid.shape)
        output = torch.cat([output_mid, output], dim=1)

        output = self.double_conv(output)

        return output

    @staticmethod
    def _same_padding(input_, target, data_format='NCHW'):#input_为same作为目标形状
        """
        Zero pad input_ as the shape of target
        """
        if data_format == 'NCHW':
            if input_.shape == target.shape:
                return input_

            else:
                assert input_.shape[:-2] == target.shape[:-2]
                _, h_input, w_input, _ = input_.shape
                _, h_target, w_target, _ = target.shape

                h_dff = h_target - h_input
                w_diff = w_target - w_input

                pad_parten = (h_dff // 2, h_dff - h_dff // 2,
                              w_diff // 2, w_diff - w_diff // 2)

                output = F.pad(input=input_, pad=pad_parten,
                               mode='constant', value=0)
                return output
        else:
            raise NotImplementedError


class FinalConv(nn.Module):
    def __init__(self, inC, outC, activate_fn='sigmod', use_bias=True,
                 outK=3, outS=1, outP=1, outD=1, outG=1):
        super(FinalConv, self).__init__()
        self.inC, self.outC = inC, outC
        self.outK, self.outS, self.outP = outK, outS, outP
        self.outD, self.outG = outD, outG
        self.use_bias = use_bias
        self.activate_fn = activate_fn

        self.conv = nn.Conv2d(in_channels=inC, out_channels=outC,
                              kernel_size=outK, stride=outS,
                              padding=outP, dilation=outD, groups=outG,
                              bias=use_bias)#输出一个参数
        if activate_fn == 'sigmoid':#激活函数
            self.act = nn.Sigmoid()

        elif activate_fn == 'relu':
            self.act = DyReLUB(outC,conv_type='2d')#nn.ReLU(inplace=True)

        else:
            raise NotImplementedError

    def forward(self, input_):
        output = self.act(self.conv(input_))
        return output