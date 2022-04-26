'''
Author: XiaoYang Xiao
'''

import cv2
import torch
import numpy as np
import torch.nn.functional as F

from apex import amp
from torch import nn
from model.resnet import get_resnet50_baseline, get_resnet101_baseline


class ASPP(nn.Module):
    def __init__(self, inchannel, out_channel, BatchNorm):
        super(ASPP, self).__init__()

        self.aspp_con1 = nn.Sequential(nn.Conv2d(inchannel, out_channel, 1, 1),
                                       BatchNorm(out_channel),
                                       nn.ReLU())
        self.aspp_con2 = nn.Sequential(nn.Conv2d(inchannel, out_channel, 3, 1, padding=6, dilation=6),
                                       BatchNorm(out_channel),
                                       nn.ReLU())
        self.aspp_con3 = nn.Sequential(nn.Conv2d(inchannel, out_channel, 3, 1, padding=12, dilation=12),
                                       BatchNorm(out_channel),
                                       nn.ReLU())
        self.aspp_con4 = nn.Sequential(nn.Conv2d(inchannel, out_channel, 3, 1, padding=18, dilation=18),
                                       BatchNorm(out_channel),
                                       nn.ReLU())
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inchannel, out_channel, 1, stride=1, bias=False),
                                             # BatchNorm(out_channel),
                                             nn.ReLU(inplace=True))
        self.out = nn.Sequential(nn.Conv2d(out_channel*5, out_channel, 1, 1),
                                 BatchNorm(out_channel),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.1))

    def forward(self, x):

        atrous_block1 = self.aspp_con1(x)
        atrous_block2 = self.aspp_con2(x)
        atrous_block3 = self.aspp_con3(x)
        atrous_block4 = self.aspp_con4(x)

        feature = self.global_avg_pool(x)
        feature = F.interpolate(feature, size=atrous_block4.size()[2:], mode='bilinear', align_corners=True)

        out = self.out(torch.cat([feature, atrous_block1, atrous_block2,
                                  atrous_block3, atrous_block4], dim=1))

        return out


class Edge_Detect(nn.Module):
    def __init__(self, in_channels, embed_dim, num_classes, BatchNorm):
        super(Edge_Detect, self).__init__()
        self.num_classes = num_classes
        self.BatchNorm = BatchNorm

        self.edge_conv1 = self.edge_conv(in_channels[0], 1)
        self.edge_conv2 = self.edge_conv(in_channels[1], 1)
        self.edge_conv3 = self.edge_conv(in_channels[2], 1)
        self.edge_conv4 = self.edge_conv(in_channels[3], num_classes)

        self.cann_conv1 = self.cann_conv(1, in_channels[0])
        self.cann_conv2 = self.cann_conv(1, in_channels[1])
        self.cann_conv3 = self.cann_conv(1, in_channels[2])

        self.edge_fuse1 = Edge_Fusion(1, BatchNorm)
        self.edge_fuse2 = Edge_Fusion(1, BatchNorm)
        self.edge_fuse3 = Edge_Fusion(1, BatchNorm)

        self.fuse = nn.Sequential(nn.Conv2d(num_classes*4, num_classes, kernel_size=1),
                                  BatchNorm(num_classes),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(0.1))

    def edge_conv(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                             self.BatchNorm(out_channels),
                             nn.ReLU(inplace=True))

    def cann_conv(self, in_channels, out_channels):
        return  nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                              self.BatchNorm(out_channels),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1),
                              self.BatchNorm(in_channels),
                              nn.ReLU(inplace=True))

    def forward(self, x):

        x1, x2, x3, x4, canny = x[0], x[1], x[2], x[3], x[4]
        
        edge1_fea = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=True)
        canny_fea = F.interpolate(canny, size=x2.size()[2:], mode='bilinear', align_corners=True)

        edge1_fea = self.edge_conv1(edge1_fea) 
        edge2_fea = self.edge_conv2(x2)
        edge3_fea = self.edge_conv3(x3)
        edge4_fea = self.edge_conv4(x4)

        canny_fea = self.cann_conv1(canny_fea)
        edge1_fea = self.edge_fuse1([edge1_fea, canny_fea])
        canny_fea = self.cann_conv2(canny_fea)
        edge2_fea = self.edge_fuse2([edge2_fea, canny_fea])
        canny_fea = self.cann_conv3(canny_fea)
        edge3_fea = self.edge_fuse3([edge3_fea, canny_fea])

        slice_ed4 = edge4_fea[:,0:1,:,:]
        fuse = torch.cat((edge1_fea, edge2_fea, edge3_fea, slice_ed4), 1)
        for i in range(edge4_fea.size(1)-1):
            slice_ed4 = edge4_fea[:, i+1:i+2, :, :]
            fuse = torch.cat((fuse, edge1_fea, edge2_fea, edge3_fea, slice_ed4), 1)

        fuse = self.fuse(fuse)

        return fuse


class Edge_Fusion(nn.Module):
    def __init__(self, in_channels, BatchNorm):
        super(Edge_Fusion, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
                                 BatchNorm(in_channels))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
                                 BatchNorm(in_channels))

        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
                                 BatchNorm(in_channels),
                                 nn.Sigmoid())
    def forward(self, x):
        edge_fea, cann_fea = x[0], x[1]

        edge_fea = self.conv_1(edge_fea)
        cann_fea = self.conv_2(cann_fea)

        psi = self.relu(edge_fea + cann_fea)
        psi = self.psi(psi) * cann_fea

        return psi


class Context_Aggregation(nn.Module):
    def __init__(self, in_channels, num_classes, middle_channels):
        super(Context_Aggregation, self).__init__()
        self.in_channels = in_channels
        self.middle_channels = middle_channels

        self.a = nn.Conv2d(in_channels, 1, 1)
        self.k = nn.Conv2d(num_classes, 1, 1)
        self.v = nn.Conv2d(in_channels, middle_channels, 1)
        self.m = nn.Conv2d(middle_channels, in_channels, 1)

    def forward(self, x, edge):
        N, C = x.size(0), self.middle_channels

        # a: [N, 1, H, W]
        a = self.a(x).sigmoid()

        # k: [N, 1, HW, 1]
        k = self.k(edge).view(N, 1, -1, 1).softmax(2)

        # v: [N, 1, C, HW]
        v = self.v(x).view(N, 1, C, -1)

        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(N, C, 1, 1)

        y = self.m(y) * a

        return x + y


class BASeg(nn.Module):
    def __init__(self, num_classes, BatchNorm=nn.BatchNorm2d, layers=50, 
                 multi_grid=(1, 1, 1), in_channels=[256, 512, 1024, 2048], depth=[2], 
                 embed_dim=512, criterion=None, pretrained=True):
        super(BASeg, self).__init__()

        self.criterion = criterion
        self.BatchNorm = BatchNorm

        if layers == 50:
            resnet = get_resnet50_baseline(pretrained=pretrained, num_classes=num_classes, BatchNorm=BatchNorm,
                                           multi_grid=multi_grid)
        elif layers == 101:
            resnet = get_resnet101_baseline(pretrained=pretrained, num_classes=num_classes, BatchNorm=BatchNorm,
                                            multi_grid=multi_grid)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
                                    resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.interpolate = F.interpolate
        del resnet

        self.aspp = ASPP(in_channels[-1], embed_dim, BatchNorm)

        self.edge_dete = Edge_Detect(in_channels, embed_dim, num_classes, BatchNorm)
        self.cont_aggr = Context_Aggregation(in_channels[-1]+embed_dim, num_classes, embed_dim)

        self.edge = nn.Sequential(
            nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False),
            self.BatchNorm(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=1))

        self.sege = nn.Sequential(
            nn.Conv2d(in_channels[-1]+embed_dim, 256, kernel_size=3, padding=1, bias=False),
            self.BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1))

        if self.training:
            self.auxc = nn.Sequential(
                nn.Conv2d(in_channels[2], 256, kernel_size=3, padding=1, bias=False),
                self.BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.),
                nn.Conv2d(256, num_classes, kernel_size=1))

        amp.register_float_function(torch, 'sigmoid')
        amp.register_float_function(torch, 'softmax')

    def down_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            self.BatchNorm(out_channels),
            nn.ReLU(inplace=True))

    def canny(self, inp, size):
        
        img = inp.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny = np.zeros((size[0], 1, size[2], size[3]))
        for i in range(size[0]):
            canny[i] = cv2.Canny(img[i], 10, 100)
        
        canny = torch.from_numpy(canny).cuda().float()
        
        return canny

    def forward(self, inp, gts=None):
        # Feature
        f0 = self.layer0(inp)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        # Size
        f0_size = inp.size()
        f1_size = f1.size()

        # Edge
        canny = self.canny(inp, f0_size)
        edge_ = self.edge_dete([f1, f2, f3, f4, canny])
        edge = self.edge(edge_)
        edge = self.interpolate(edge, f0_size[2:], mode='bilinear', align_corners=True)

        # ASPP
        x = torch.cat([self.aspp(f4), f4], dim=1)

        # Context Aggregation
        x = self.cont_aggr(x, edge_)

        # Seg
        x = self.sege(x)
        x = self.interpolate(x, f0_size[2:], mode='bilinear', align_corners=True)
        
        # Loss
        if self.training:
            aux = self.auxc(f3)
            aux = self.interpolate(aux, size=f0_size[2:], mode='bilinear', align_corners=True)
            return x, edge, self.criterion((x, aux, edge), gts)
        else:
            return x, edge
