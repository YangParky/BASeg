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


class PositionAttentionModule(nn.Module):
    """ Position attention module"""
    def __init__(self, in_channels, **kwargs):
        super(PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, _, H, W = x.size()
        feat_b = self.conv_b(x).view(B, -1, H * W).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(B, -1, H * W)

        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(B, -1, H * W)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(B, -1, H, W)

        out = self.alpha * feat_e + x

        return out


class ChannelAttentionModule(nn.Module):
    """Channel attention module"""
    def __init__(self, **kwargs):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)

        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


class DAHead(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(DAHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = PositionAttentionModule(inter_channels, **kwargs)
        self.cam = ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        return feat_fusion


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = in_channels // 2
        self.conv_phi = nn.Conv2d(in_channels=in_channels, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=in_channels, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=in_channels, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c//2, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c//2, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c//2, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)

        out = mask + x

        return out


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_channels):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width))\
            .view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        return self.gamma * (out_H + out_W) + x


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



class Edge_Detect(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Edge_Detect, self).__init__()

        self.edge_conv1 = self.edge_conv(in_channels[0], 1)
        self.edge_conv2 = self.edge_conv(in_channels[1], 1)
        self.edge_conv3 = self.edge_conv(in_channels[2], 1)
        self.edge_conv4 = self.edge_conv(in_channels[3], out_channels)

        self.cann_conv1 = self.cann_conv(1, in_channels[0])
        self.cann_conv2 = self.cann_conv(1, in_channels[1])
        self.cann_conv3 = self.cann_conv(1, in_channels[2])

        self.edge_fuse1 = Edge_Fusion(1)
        self.edge_fuse2 = Edge_Fusion(1)
        self.edge_fuse3 = Edge_Fusion(1)

        self.fuse = nn.Sequential(nn.Conv2d(out_channels * 4, out_channels, kernel_size=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(0.1))

    def edge_conv(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))

    def cann_conv(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1),
                             nn.BatchNorm2d(in_channels),
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

        slice_ed4 = edge4_fea[:, 0:1, :, :]
        fuse = torch.cat((edge1_fea, edge2_fea, edge3_fea, slice_ed4), 1)
        for i in range(edge4_fea.size(1) - 1):
            slice_ed4 = edge4_fea[:, i + 1:i + 2, :, :]
            fuse = torch.cat((fuse, edge1_fea, edge2_fea, edge3_fea, slice_ed4), 1)

        fuse = self.fuse(fuse)

        return fuse


class Edge_Fusion(nn.Module):
    def __init__(self, in_channels):
        super(Edge_Fusion, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
                                    nn.BatchNorm2d(in_channels))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
                                    nn.BatchNorm2d(in_channels))

        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
                                 nn.BatchNorm2d(in_channels),
                                 nn.Sigmoid())

    def forward(self, x):
        edge_fea, cann_fea = x[0], x[1]

        edge_fea = self.conv_1(edge_fea)
        cann_fea = self.conv_2(cann_fea)

        psi = self.relu(edge_fea + cann_fea)
        psi = self.psi(psi) * cann_fea

        return psi


class Context_Aggregation(nn.Module):
    def __init__(self, in_channels, edge_channels, middle_channels):
        super(Context_Aggregation, self).__init__()
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
        # self.da_head = DAHead(in_channels=in_channels[-1])
        # self.pr_head = nn.Sequential(
        #     nn.Conv2d(in_channels[-1], embed_dim, kernel_size=1, padding=0, bias=False),
        #     nn.BatchNorm2d(embed_dim))
        # self.cc_head = CrissCrossAttention(in_channels=embed_dim)
        # self.nl_head = NonLocalBlock(in_channels=embed_dim)
        self.as_head = ASPP(in_channels=in_channels[-1], out_channels=embed_dim)

        self.conv_sege = nn.Sequential(
            nn.Conv2d(in_channels[-1]+embed_dim, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1))

        self.edge_detect = Edge_Detect(in_channels=in_channels, mid_channels=128, out_channels=num_classes)
        self.cont_aggreg = Context_Aggregation(in_channels=in_channels[-1]+embed_dim, edge_channels=num_classes, middle_channels=embed_dim)
        self.edge_sege = nn.Sequential(nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(1, 1, kernel_size=1))

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

        # Edge
        canny = self.canny(inp, f0_size)
        edge_ = self.edge_detect([f1, f2, f3, f4, canny])
        edge = self.edge_sege(edge_)
        edge = self.interpolate(edge, f0_size[2:], mode='bilinear', align_corners=True)

        # ASPP
        # x = self.da_head(f4)
        # x = self.pr_head(f4)
        # x = self.cc_head(x)
        # x = self.nl_head(x)
        # x = self.cm_head(x)
        x = self.as_head(f4)
        x = torch.cat([x, f4], dim=1)

        # Context Aggregation
        x = self.cont_aggreg(x, edge_)

        # Seg
        x = self.conv_sege(x)
        x = self.interpolate(x, f0_size[2:], mode='bilinear', align_corners=True)

        # Loss
        # if self.training:
            aux = self.aux(f3)
            aux = self.interpolate(aux, size=f0_size[2:], mode='bilinear', align_corners=True)
            return x, edge, self.criterion((x, aux, edge), gts)
        else:
            return x, edge
