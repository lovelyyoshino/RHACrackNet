import torch.nn as nn
import torch
import torch.nn.functional as F
from .batchnorm import SynchronizedBatchNorm2d
import math

BatchNorm2d = SynchronizedBatchNorm2d

class module_init(nn.Module):
    def __init__(self, inplanes, planes,stride = 2):
        super(module_init, self).__init__()

        #Entry flow
        self.conv1 = nn.Conv2d(inplanes, 32, 3, stride=stride, padding=stride//2, bias=False)
        self.bn1 = BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x




class SeparableConv2d_same(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv2d_same, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


    
class Block_1(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, stride = 2):
        super(Block_1, self).__init__()
        self.conv1 = SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = SeparableConv2d_same(planes, planes, 3, stride=1, dilation=dilation)
        self.bn2 = BatchNorm2d(planes)
        self. conv3 = SeparableConv2d_same(planes, planes, 3, stride=stride, dilation=dilation) 
        self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, padding = 0, bias=False)
        self.bn3 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x_1 = self.relu(self.bn1(self.conv1(x))) 
        x_1 = self.bn2(self.conv2(x_1))
        x_1 = self.conv3(x_1)
        skip = self.skip(x)
        skip = self.bn3(skip)
        out = skip+x_1
        out = self.relu(out)
        return out 

class Block_2(nn.Module):
    def __init__(self, inplanes, planes, dilation=1):
        super(Block_2, self).__init__()
        self.conv1 = SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = SeparableConv2d_same(planes, planes, 3, stride=1, dilation=dilation)
        self.bn2 = BatchNorm2d(planes)
        self. conv3 = SeparableConv2d_same(planes, planes, 3, stride=1, dilation=dilation) 
        self.bn3 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_1 = self.relu(self.bn1(self.conv1(x)))
        x_1 = self.relu(self.bn2(self.conv2(x_1)))
        x_1 = self.bn3(self.conv3(x_1))
        out = x+x_1
        out = self.relu(out)
        
        return out 

class module_fuse(nn.Module):
    def __init__(self, planes_1, planes_2, planes_3, planes_4, planes_5, planes_out):
        super(module_fuse, self).__init__()

        #Entry flow
        self.conv1 = nn.Conv2d(planes_1, planes_out, 1, 1, 0, 1, 1)
#        self.convTrans1 = nn.ConvTranspose2d(planes_out, planes_out, 3, stride=2, dilation=1)
        
        self.conv2 = nn.Conv2d(planes_2, planes_out, 1, 1, 0, 1, 1)
#        self.convTrans2 = SeparableConvTranspose2d_same(planes_out, planes_out, 3, stride=4, dilation=1)
        
        self.conv3 = nn.Conv2d(planes_3, planes_out, 1, 1, 0, 1, 1)
#        self.convTrans3 = SeparableConvTranspose2d_same(planes_out, planes_out, 3, stride=8, dilation=1)
        
        self.conv4 = nn.Conv2d(planes_4, planes_out, 1, 1, 0, 1, 1)
#        self.convTrans4 = SeparableConvTranspose2d_same(planes_out, planes_out, 3, stride=16, dilation=1)
        
        self.conv5 = nn.Conv2d(planes_5, planes_out, 1, 1, 0, 1, 1)
#        self.convTrans5 = SeparableConvTranspose2d_same(planes_out, planes_out, 3, stride=32, dilation=1)

    def forward(self, x_1, x_2, x_3, x_4, x_5):
        x_1 = self.conv1(x_1)
        x_1 = F.interpolate(x_1, size=[320,480], mode='bilinear', align_corners=True)
        x_2 = self.conv2(x_2)
        x_2 = F.interpolate(x_2, size=[320,480], mode='bilinear', align_corners=True)
        x_3 = self.conv3(x_3)
        x_3 = F.interpolate(x_3, size=[320,480], mode='bilinear', align_corners=True)
        x_4 = self.conv4(x_4)
        x_4 = F.interpolate(x_4, size=[320,480], mode='bilinear', align_corners=True)
        x_5 = self.conv5(x_5)
        x_5 = F.interpolate(x_5, size=[320,480], mode='bilinear', align_corners=True)
        

        out = torch.cat((x_1, x_2, x_3, x_4, x_5),dim=1)
        return out
    
class module_end(nn.Module):
    def __init__(self, inplanes, planes):
        super(module_end, self).__init__()

        #Entry flow
        self.conv1 = nn.Conv2d(inplanes, 32, 3, stride=1, padding = 1)
        self.bn1 = BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_1 = nn.Dropout2d(0.5)
        
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding = 1)
        self.bn2 = BatchNorm2d(32)
        self.dropout_2 = nn.Dropout2d(0.1)
        
        self.conv3 = nn.Conv2d(32, planes, 3, stride=1, padding = 1)
        self.sigmoid = nn.Sigmoid()
        


    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout_1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout_2(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x

class ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(ASPPModule, self).__init__()
        if dilation == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = dilation
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
class ASPP(nn.Module):
    def __init__(self, inplanes, planes):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]
       
        self.aspp1 = ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             BatchNorm2d(256),
                                             nn.ReLU(inplace=True))
        
        self.conv1 = nn.Conv2d(1024, planes, 1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        self._init_weight()
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

#        x5 = self.global_avg_pool(x)
#        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
class Decoder(nn.Module):
    def __init__(self, low_level_inplanes, num_classes):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat):

        low_level_feat = self.relu(low_level_feat)
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)

        x = self.last_conv(x)

        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
 
def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

class SeparableConvTranspose2d_same(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConvTranspose2d_same, self).__init__()

        self.conv1 = nn.ConvTranspose2d(inplanes, inplanes, kernel_size, stride, 1, dilation,
                               groups=1, bias=bias)
#        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
#        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
#        x = self.pointwise(x)
        return x          
    
if __name__ == '__main__':
    model = Block_1(128,64,stride=2)
    model.eval()
    image_1 = torch.randn(1, 128, 30, 40)

    with torch.no_grad():
        output = model.forward(image_1)
    print(output.size())