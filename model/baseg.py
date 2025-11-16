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
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        self.aspp_con1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, 1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU())
        self.aspp_con2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=6, dilation=6),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU())
        self.aspp_con3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=12, dilation=12),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU())
        self.aspp_con4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=18, dilation=18),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU())
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
                                             nn.ReLU(inplace=True))
        self.out = nn.Sequential(nn.Conv2d(out_channels*5, out_channels, 1, 1),
                                 nn.BatchNorm2d(out_channels),
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


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Boundary Refine Module
class BRM(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None):
        super(BRM, self).__init__()

        self.edge_conv1 = self.edge_conv(in_channels[0], 1)
        self.edge_conv2 = self.edge_conv(in_channels[1], 1)
        self.edge_conv3 = self.edge_conv(in_channels[2], 1)
        self.edge_conv4 = self.edge_conv(in_channels[3], out_channels)

        self.canny_conv1 = self.canny_conv(1, in_channels[0])
        self.canny_conv2 = self.canny_conv(1, in_channels[1])
        self.canny_conv3 = self.canny_conv(1, in_channels[2])

        self.edge_agb1 = AGB(1)
        self.edge_agb2 = AGB(1)
        self.edge_agb3 = AGB(1)

        self.fuse = nn.Sequential(nn.Conv2d(out_channels * 4, out_channels, kernel_size=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(0.1))

    def edge_conv(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))

    def canny_conv(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1),
                             nn.BatchNorm2d(in_channels),
                             nn.ReLU(inplace=True))

    def forward(self, f1, f2, f3, f4, canny):

        edge1_feat = F.interpolate(f1, size=f2.size()[2:], mode='bilinear', align_corners=True)
        canny_feat = F.interpolate(canny, size=f2.size()[2:], mode='bilinear', align_corners=True)

        edge1_feat = self.edge_conv1(edge1_feat)
        edge2_feat = self.edge_conv2(f2)
        edge3_feat = self.edge_conv3(f3)
        edge4_feat = self.edge_conv4(f4)

        canny_feat = self.cann_conv1(canny_feat)
        edge1_feat = self.edge_agb1([edge1_feat, canny_feat])
        canny_feat = self.cann_conv2(canny_feat)
        edge2_feat = self.edge_agb2([edge2_feat, canny_feat])
        canny_feat = self.cann_conv3(canny_feat)
        edge3_feat = self.edge_agb3([edge3_feat, canny_feat])

        slice_ed4 = edge4_feat[:, 0:1, :, :]
        fuse = torch.cat((edge1_feat, edge2_feat, edge3_feat, slice_ed4), 1)
        for i in range(edge4_feat.size(1) - 1):
            slice_ed4 = edge4_feat[:, i + 1:i + 2, :, :]
            fuse = torch.cat((fuse, edge1_feat, edge2_feat, edge3_feat, slice_ed4), 1)

        fuse = self.fuse(fuse)

        return fuse


# Attention Gate Block
class AGB(nn.Module):
    def __init__(self, in_channels):
        super(AGB, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
                                    nn.BatchNorm2d(in_channels))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
                                    nn.BatchNorm2d(in_channels))

        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
                                 nn.BatchNorm2d(in_channels),
                                 nn.Sigmoid())

    def forward(self, x):
        edge_feat, canny_feat = x[0], x[1]

        edge_feat = self.conv_1(edge_feat)
        canny_feat = self.conv_2(canny_feat)

        psi = self.relu(edge_feat + canny_feat)
        psi = self.psi(psi) * canny_feat

        return psi


# Context Aggregation Module
class CAM(nn.Module):
    def __init__(self, in_channels, edge_channels, middle_channels):
        super(CAM, self).__init__()
        self.in_channels = in_channels
        self.middle_channels = middle_channels

        self.a = nn.Conv2d(in_channels, 1, 1)
        self.k = nn.Conv2d(edge_channels, 1, 1)
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
    def __init__(self, num_classes, layers=50, multi_grid=(1, 1, 1), in_channels=[256, 512, 1024, 2048],
                 embed_dim=512, criterion=None, pretrained=True):
        super(BASeg, self).__init__()
        self.criterion = criterion

        if layers == 50:
            resnet = get_resnet50_baseline(pretrained=pretrained, num_classes=num_classes, multi_grid=multi_grid)
        elif layers == 101:
            resnet = get_resnet101_baseline(pretrained=pretrained, num_classes=num_classes, multi_grid=multi_grid)
        else:
            resnet = None

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
                                    resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.interpolate = F.interpolate
        del resnet

        # 2048 -> 512
        self.ASPP = ASPP(in_channels=embed_dim, out_channels=embed_dim)
        self.BRM = BRM(in_channels=in_channels, mid_channels=128, out_channels=num_classes)
        self.CAM = CAM(in_channels=in_channels[-1], edge_channels=num_classes, middle_channels=embed_dim)

        self.edge_seg = nn.Sequential(nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(1, 1, kernel_size=1))
        self.conv_seg = nn.Sequential(
            nn.Conv2d(in_channels[-1], 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1))

        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(in_channels[2], 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.),
                nn.Conv2d(256, num_classes, kernel_size=1))

        amp.register_float_function(torch, 'sigmoid')
        amp.register_float_function(torch, 'softmax')

    def down_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def get_canny_feature(self, inp, size):
        
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

        # ASPP
        x = self.ASPP(f4)
        x = torch.cat([x, f4], dim=1)

        # Edge
        canny = self.get_canny_feature(inp, f0_size)
        edge_ = self.BRM(f1, f2, f3, f4, canny)

        # Context Aggregation
        x = self.CAM(x, edge_)

        # Seg
        x = self.conv_seg(x)
        x = self.interpolate(x, f0_size[2:], mode='bilinear', align_corners=True)
        edge = self.edge_seg(edge_)
        edge = self.interpolate(edge, f0_size[2:], mode='bilinear', align_corners=True)

        # Loss
        if self.training:
            aux = self.aux(f3)
            aux = self.interpolate(aux, size=f0_size[2:], mode='bilinear', align_corners=True)
            return x, edge, self.criterion((x, aux, edge), gts)
        else:
            return x, edge
