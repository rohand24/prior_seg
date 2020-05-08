from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import torchvision


import pdb


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class custom_conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, one_D=False):
        super(custom_conv_block, self).__init__()
        if not one_D:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=stride, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=kernel, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True))
            

    def forward(self, x):

        x = self.conv(x)
        return x

class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 *32, n1*64, n1 * 128]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool7 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        self.Conv6 = conv_block(filters[4], filters[5])
        self.Conv7 = conv_block(filters[5], filters[6])
#         self.Conv8 = conv_block(filters[6], filters[7])

#         self.Up8 = up_conv(filters[7], filters[6])
#         self.Up_conv8 = conv_block(filters[7] , filters[6])
        
        self.Up7 = up_conv(filters[6], filters[5])
        self.Up_conv7 = conv_block(filters[6] , filters[5])

        self.Up6 = up_conv(filters[5], filters[4])
        self.Up_conv6 = conv_block(filters[5] , filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

#         self.active = torch.sigmoid()

    def forward(self, x):
        
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        
        e6 = self.Maxpool5(e5)
        e6 = self.Conv6(e6)
        
        e7 = self.Maxpool6(e6)
        e7 = self.Conv7(e7)
        
#         e8 = self.Maxpool7(e7)
#         e8 = self.Conv8(e8)

#         d8 = self.Up8(e8)
#         d8 = torch.cat((e7, d8), dim=1)
#         d8 = self.Up_conv8(d8)
        
        d7 = self.Up7(e7)
        d7 = torch.cat((e6, d7), dim=1)
        d7 = self.Up_conv7(d7)

        d6 = self.Up6(e6)
        d6 = torch.cat((e5, d6), dim=1)
        d6 = self.Up_conv6(d6)
        
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv(d2)

        out = torch.sigmoid(d1)

        return out


class custom_unet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(custom_unet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 6, n1 * 8, n1 * 10, n1 * 12, n1 * 16 ]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        self.Conv6 = conv_block(filters[4], filters[5])
        self.Conv7 = custom_conv_block(filters[5], filters[6])
        self.Conv8 = custom_conv_block(filters[6], filters[7], one_D=True)

        self.Up8 = up_conv(filters[7], filters[6])
        self.Up_conv8 = conv_block(filters[6] * 2 , filters[6])
        
        self.Up7 = up_conv(filters[6], filters[5])
        self.Up_conv7 = conv_block(filters[5] * 2 , filters[5])

        self.Up6 = up_conv(filters[5], filters[4])
        self.Up_conv6 = conv_block(filters[4] * 2 , filters[4])
        
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[3] * 2 , filters[3])

#         self.Up4 = up_conv(filters[3], filters[2])
#         self.Up_conv4 = conv_block(filters[3], filters[2])

#         self.Up3 = up_conv(filters[2], filters[1])
#         self.Up_conv3 = conv_block(filters[2], filters[1])

#         self.Up2 = up_conv(filters[1], filters[0])
#         self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[3], out_ch, kernel_size=1, stride=1, padding=0)

#         self.active = torch.sigmoid()

    def forward(self, x):
        
        
        e1 = self.Conv1(x)
        e1 = self.Maxpool1(e1)
        
        e2 = self.Conv2(e1)
        e2 = self.Maxpool1(e2)
        
        e3 = self.Conv3(e2)
        e3 = self.Maxpool1(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        
        e6 = self.Maxpool5(e5)
        e6 = self.Conv6(e6)
        
        e7 = self.Maxpool6(e6)
        e7 = self.Conv7(e7)

#         e8 = self.Maxpool6(e7)
#         e8 = e8.squeeze(dim=3)
#         e8 = self.Conv8(e8)
#         e8 = e8.unsqueeze(dim=3)

# #         pdb.set_trace()
#         d8 = self.Up8(e8)
#         d8 = torch.cat((e7, d8), dim=1)
#         d8 = self.Up_conv8(d8)
        
        d7 = self.Up7(e7)
        d7 = torch.cat((e6, d7), dim=1)
        d7 = self.Up_conv7(d7)

        d6 = self.Up6(e6)
        d6 = torch.cat((e5, d6), dim=1)
        d6 = self.Up_conv6(d6)
        
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

#         d4 = self.Up4(d5)
#         d4 = torch.cat((e3, d4), dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         d3 = torch.cat((e2, d3), dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         d2 = torch.cat((e1, d2), dim=1)
#         d2 = self.Up_conv2(d2)

        d1 = self.Conv(d5)

        out = torch.sigmoid(d1)

        return out
    


#For nested 3 channels are required

class conv_block_nested(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output
    
#Nested Unet

class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(NestedUNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)


    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output
 
    
# Deep Nested Unet

class DeepNestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(DeepNestedUNet, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.conv5_0 = conv_block_nested(filters[4], filters[5], filters[5])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])
        self.conv4_1 = conv_block_nested(filters[4] + filters[5], filters[4], filters[4])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])
        self.conv3_2 = conv_block_nested(filters[3]*2 + filters[4], filters[3], filters[3])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])
        self.conv2_3 = conv_block_nested(filters[2]*3 + filters[3], filters[2], filters[2])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])
        self.conv1_4 = conv_block_nested(filters[1]*4 + filters[2], filters[1], filters[1])
        
        self.conv0_5 = conv_block_nested(filters[0]*5 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)


    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))
        
        x5_0 = self.conv5_0(self.pool(x4_0))
        x4_1 = self.conv4_1(torch.cat([x4_0, self.Up(x5_0)], 1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.Up(x4_1)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.Up(x3_2)], 1))
        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.Up(x2_3)], 1))
        x0_5 = self.conv0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.Up(x1_4)], 1))

        output = self.final(x0_5)
        return output

    
class RohanJNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(RohanJNet, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1*32]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.conv5_0 = conv_block_nested(filters[4], filters[5], filters[5])

#         self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])
        self.conv4_1 = conv_block_nested(filters[4] + filters[5], filters[4], filters[4])

#         self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])
        self.conv3_2 = conv_block_nested(filters[3]*2 + filters[4], filters[3], filters[3])

#         self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])
        self.conv2_3 = conv_block_nested(filters[2]*3 + filters[3], filters[2], filters[2])

#         self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])
        self.conv1_4 = conv_block_nested(filters[1]*4 + filters[2], filters[1], filters[1])

        self.final = nn.Conv2d(filters[1], out_ch, kernel_size=1)


    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
#         x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
#         x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
#         x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
#         x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))
        
        x5_0 = self.conv5_0(self.pool(x4_0))
        x4_1 = self.conv4_1(torch.cat([x4_0, self.Up(x5_0)], 1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.Up(x4_1)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.Up(x3_2)], 1))
        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.Up(x2_3)], 1))
    
        output = self.final(x1_4)
        return output
    
    
    
    
    
    
    
#Dictioary Unet
#if required for getting the filters and model parameters for each step 

class ConvolutionBlock(nn.Module):
    """Convolution block"""

    def __init__(self, in_filters, out_filters, kernel_size=3, batchnorm=True, last_active=F.relu):
        super(ConvolutionBlock, self).__init__()

        self.bn = batchnorm
        self.last_active = last_active
        self.c1 = nn.Conv2d(in_filters, out_filters, kernel_size, padding=1)
        self.b1 = nn.BatchNorm2d(out_filters)
        self.c2 = nn.Conv2d(out_filters, out_filters, kernel_size, padding=1)
        self.b2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.c1(x)
        if self.bn:
            x = self.b1(x)
        x = F.relu(x)
        x = self.c2(x)
        if self.bn:
            x = self.b2(x)
        x = self.last_active(x)
        return x

    
    
class custom_NestedUnet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(custom_NestedUnet, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 8, n1 *4]
#         ipdb.set_trace()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.conv5_0 = conv_block_nested(filters[4], filters[5], filters[5])
        self.conv6_0 = conv_block_nested(filters[5]+1, filters[6], filters[6])
        
#         self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
#         self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])
        self.conv4_1 = conv_block_nested(filters[4] + filters[5], filters[4], filters[4])
        self.conv5_1 = conv_block_nested(filters[5] + filters[6]+1, filters[5], filters[5])

#         self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
#         self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])
        self.conv3_2 = conv_block_nested(filters[2] + filters[3]*2 + filters[4], filters[3], filters[3])
        self.conv4_2 = conv_block_nested(filters[3] + filters[4]*2 + filters[5], filters[4], filters[4])

#         self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
#         self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])
#         self.conv2_3 = conv_block_nested(filters[2]*3 + filters[3], filters[2], filters[2])
#         self.conv3_3 = conv_block_nested(filters[3]*3 + filters[4], filters[3], filters[3])

#         self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])
#         self.conv1_4 = conv_block_nested(filters[1]*4 + filters[2], filters[1], filters[1])
#         self.conv2_4 = conv_block_nested(filters[2]*4 + filters[3], filters[2], filters[2])
        
#         self.conv0_5 = conv_block_nested(filters[0]*5 + filters[1], filters[0], filters[0])
#         self.conv1_5 = conv_block_nested(filters[1]*5 + filters[2], filters[1], filters[1])
        
#         self.conv0_6 = conv_block_nested(filters[0]*6 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[4], out_ch, kernel_size=1)


    def forward(self, x):
        
        
        im64 = self.pool(self.pool(x))[:,1,:,:].unsqueeze(1)
        im16 = self.pool(self.pool(im64))
        im8 = self.pool(im16)
        im4 = self.pool(im8)
        im2 = self.pool(im4)
        
#         tmask = x[:,1,:,:]
#         tmask
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
#         x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
#         x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
#         x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
#         x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
#         x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
#         x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
#         x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))
        
        x5_0 = self.conv5_0(self.pool(x4_0))
        x4_1 = self.conv4_1(torch.cat([x4_0, self.Up(x5_0)], 1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.Up(x4_1), self.pool(x2_2)], 1))
#         x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.Up(x3_2)],1))
#         x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.Up(x2_3)],1))
#         x0_5 = self.conv0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.Up(x1_4)],1))
        
#         pdb.set_trace()
        x6_0 = self.conv6_0(torch.cat([self.pool(x5_0), im4],1))
        x5_1 = self.conv5_1(torch.cat([x5_0,self.Up(x6_0), im8],1))
        x4_2 = self.conv4_2(torch.cat([x4_0, x4_1, self.Up(x5_1), self.pool(x3_2)], 1))
#         x3_3 = self.conv3_3(torch.cat([x3_0, x3_1, x3_2, self.Up(x4_2)], 1))
#         x2_4 = self.conv2_4(torch.cat([x2_0, x2_1, x2_2, x2_3, self.Up(x3_3)],1))
#         x1_5 = self.conv1_5(torch.cat([x1_0, x1_1, x1_2, x1_3, x1_4, self.Up(x2_3)],1))
#         x0_6 = self.conv0_6(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, x0_5, self.Up(x1_5)],1))

        output = self.final(x4_2)
        return output


class custom_conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, one_D=False):
        super(custom_conv_block, self).__init__()
        if not one_D:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=stride, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=kernel, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True))
            

    def forward(self, x):
        
        x = self.conv(x)
        return x
# custom nested Unet with lowest layer 2x2 and out size 16
    
class CustomNestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(CustomNestedUNet, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 8, n1 *4, n1*2]
#         ipdb.set_trace()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.conv5_0 = conv_block_nested(filters[4], filters[5], filters[5])
        self.conv6_0 = conv_block_nested(filters[5], filters[6], filters[6])
        self.conv7_0 = custom_conv_block(filters[6] +1 , filters[4], one_D=True)
        
#         self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
#         self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
#         self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
#         self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])
        self.conv4_1 = conv_block_nested(filters[4] + filters[5], filters[4], filters[4])
        self.conv5_1 = conv_block_nested(filters[5] + filters[6], filters[5], filters[5])
        self.conv6_1 = conv_block_nested(filters[6] + filters[4] +1 , filters[6], filters[6])

#         self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
#         self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
#         self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])
#         self.conv3_2 = conv_block_nested(filters[3]*2 + filters[4], filters[3], filters[3]) # filters[2] + 
        self.conv4_2 = conv_block_nested(filters[4]*2 + filters[5], filters[4], filters[4]) # filters[3] + 
        self.conv5_2 = conv_block_nested(filters[5]*2 + filters[6] +1 , filters[4], filters[5]) # filters[4] + 

#         self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
#         self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])
#         self.conv2_3 = conv_block_nested(filters[2]*3 + filters[3], filters[2], filters[2])
#         self.conv3_3 = conv_block_nested(filters[2]+ filters[3]*3 + filters[4], filters[3], filters[3])
        self.conv4_3 = conv_block_nested(filters[4]*3 + filters[5] +1 , filters[4], filters[4])

#         self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])
#         self.conv1_4 = conv_block_nested(filters[1]*4 + filters[2], filters[1], filters[1])
#         self.conv2_4 = conv_block_nested(filters[2]*4 + filters[3], filters[2], filters[2])
        
#         self.conv0_5 = conv_block_nested(filters[0]*5 + filters[1], filters[0], filters[0])
#         self.conv1_5 = conv_block_nested(filters[1]*5 + filters[2], filters[1], filters[1])
        
#         self.conv0_6 = conv_block_nested(filters[0]*6 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[4], out_ch, kernel_size=1)


    def forward(self, x):
#         ipdb.set_trace()
        
        im64 = self.pool(self.pool(x))[:,1,:,:].unsqueeze(1)
        im16 = self.pool(self.pool(im64))
        im8 = self.pool(im16)
        im4 = self.pool(im8)
        im2 = self.pool(im4)
        
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
#         x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
#         x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
#         x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
#         x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
#         x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
#         x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
#         x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
#         x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
#         x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
#         x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))
        
        x5_0 = self.conv5_0(self.pool(x4_0))
        x4_1 = self.conv4_1(torch.cat([x4_0, self.Up(x5_0)], 1))
#         x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.Up(x4_1)], 1))
#         x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.Up(x3_2)],1))
#         x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.Up(x2_3)],1))
#         x0_5 = self.conv0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.Up(x1_4)],1))
        
        x6_0 = self.conv6_0(self.pool(x5_0))
        x5_1 = self.conv5_1(torch.cat([x5_0,self.Up(x6_0)],1))
        x4_2 = self.conv4_2(torch.cat([x4_0, x4_1, self.Up(x5_1)], 1))
#         x3_3 = self.conv3_3(torch.cat([x3_0, x3_1, x3_2, self.Up(x4_2), self.pool(x2_3)], 1))
#         x2_4 = self.conv2_4(torch.cat([x2_0, x2_1, x2_2, x2_3, self.Up(x3_3)],1))
#         x1_5 = self.conv1_5(torch.cat([x1_0, x1_1, x1_2, x1_3, x1_4, self.Up(x2_3)],1))
#         x0_6 = self.conv0_6(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, x0_5, self.Up(x1_5)],1))
#         pdb.set_trace()
        x7_0 = self.conv7_0(torch.cat([self.pool(x6_0), im2],1))
        x6_1 = self.conv6_1(torch.cat([x6_0,self.Up(x7_0), im4],1))
        x5_2 = self.conv5_2(torch.cat([x5_0, x5_1, self.Up(x6_1), im8], 1))
        x4_3 = self.conv4_3(torch.cat([x4_0, x4_1, x4_2, self.Up(x5_2), im16], 1))

        output = self.final(x4_3)
        return output

    
