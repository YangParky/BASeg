#
# #
# # visual feature map
# #
#
# import os
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
#
# from PIL import Image
# from torchvision import transforms
# from model.baseg import BASeg
#
# input_data = []
# output_data = []
# activation = {}
#
#
# def get_gradient(model, inputs, outputs):
#     input_data.append(inputs)
#     output_data.append(outputs)
#
#
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook
#
#
# def compute_dpixel(src_x, src_w, dst_w):
#     scale_x = src_w / dst_w
#     shift = 0.5 * (scale_x - 1)
#     dst_x = (src_x - shift) * dst_w / src_w
#
#     return dst_x
#
#
# def compute_rpixel(dst_x, src_w, dst_w):
#     scale_x = src_w / dst_w
#     shift = 0.5 * (scale_x - 1)
#     src_x = scale_x * dst_x + shift
#
#     return src_x
#
#
# # model_path = './exp/cityscapes/attenet101/model/baseline_aspp_canny_gate_lre2_flohem_tv/train_epoch_200.pth'
# # image_path = '../../Dataset/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_002196_leftImg8bit.png'
# model_path = './exp/ade20k/attenet101/model/baseline_aspp_canny_gate_lr5e3/train_epoch_150.pth'
# image_path = '../../Dataset/ADE20K/Scene-Parsing/ADEChallengeData2016/images/validation/ADE_val_00000174.jpg'
#
# model = BASeg(layers=101, num_classes=19, in_channels=[256, 512, 1024, 2048], embed_dim=512,
#               depth=[1], multi_grid=tuple([1, 1, 1]), pretrained=True)
#
# checkpoint = torch.load(model_path)
# model.load_state_dict(checkpoint['state_dict'], strict=False)
# model = model.to(torch.device('cuda:0'))
#
# transform = transforms.Compose([
#     transforms.Resize([520, 520]),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
#
# img = Image.open(image_path)
# img = transform(img).unsqueeze(0).cuda()
#
# model.eval()
# model.cont_aggreg.register_forward_hook(get_gradient)
# model.layer1.register_forward_hook(get_activation('Context_Aggregation'))
# result = model(img)
#
# cont_aggreg_1 = output_data[0]
# cont_aggreg_2 = activation['Context_Aggregation']
#
# print(compute_rpixel(350, cont_aggreg_2.shape[-1], result[0].shape[-1]))
# print(compute_rpixel(500, cont_aggreg_2.shape[-1], result[0].shape[-1]))
# print(compute_dpixel(220, cont_aggreg_2.shape[-1], result[0].shape[-1]))
# print(compute_dpixel(250, cont_aggreg_2.shape[-1], result[0].shape[-1]))
#
# point = cont_aggreg_2
# point = point[:, :, 100, 125].unsqueeze(2).unsqueeze(3)
# point = point.expand(point.size())
#
# visual = torch.cosine_similarity(point, cont_aggreg_2, dim=1)
# visual = visual.unsqueeze(1)
# visual = F.interpolate(visual, size=result[0].size()[2:], mode='bilinear', align_corners=True)
#
# plt.figure(figsize=(12, 12))
# plt.imshow(visual.cpu().detach().numpy()[0, 0, :, :], cmap='jet')
# plt.axis('off')
# plt.savefig('./exp/cityscapes/baseg101/result/baseline_aspp_canny_gate_lre2_flohem_tv/ms/validation/visual/point' + '.png', dpi=100, bbox_inches='tight')
#
#
#
# #
# # plot error maps
# #
# import os
# import cv2
# import numpy as np
#
# from PIL import Image, ImageChops
#
# predi_pa = './exp/cityscapes/attenet101/result/baseline_aspp_canny_gate_lre2_flohem_tv/ms/validation/visual/'
# predi_pa = '../../Project/mmsegmentation/show_dirs/psanet_r101_d8_769x769_80k_cityscapes/frankfurt/'
# truth_pa = '../../Dataset/cityscapes/gtFine/val/frankfurt/'
#
# name_pred = 'frankfurt_000001_003056_leftImg8bit.png'
# name_trut = 'frankfurt_000001_003056_gtFine_color.png'
#
# truth_img = cv2.imread(truth_pa+name_trut).astype('uint8')
# predi_img = cv2.imread(predi_pa+name_pred).astype('uint8')
# predi_img[np.where(truth_img == 0)] = 0
#
# diff = cv2.absdiff(predi_img, truth_img)
# diff = 255 - diff
#
# cv2.imwrite('./exp/cityscapes/attenet101/result/baseline_aspp_canny_gate_lre2_flohem_tv/ms/validation/visual/error.png', diff)
# cv2.imshow('diff', diff)
# cv2.waitKey(1000000)
#
#
#
# #
# # make a video
# #
# import os
# import cv2
#
# fps = 60
# fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
#
# dat_path = './data/cityscapes/list/demoVideo.txt'
# img_path = './exp/cityscapes/baseg101/video/baseline_aspp_canny_gate_lre3_floss/ss/color/'
# vid_path = './exp/cityscapes/baseg101/video/baseline_aspp_canny_gate_lre3_floss/ss/convert/color2.avi'
#
# videoWriter = cv2.VideoWriter(vid_path, fourcc, fps, (2048, 1024))
# img_name = []
#
# with open(dat_path, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.strip()
#         line = line.split('/')[-1]
#         img_name.append(line)
#
# for i in range(len(img_name)):
#     img = cv2.imread(img_path+img_name[i])
#     videoWriter.write(img)
#
# videoWriter.release()
# print('finished~')
#
#
#
# #
# # color ade20k gt
# #
# import os
# import cv2
#
# from PIL import Image
#
# adepallete = [
# 	0, 0, 0, 120, 120, 120, 180, 120, 120, 6, 230, 230, 80, 50, 50, 4, 200, 3, 120, 120, 80, 140, 140, 140, 204,
# 	5, 255, 230, 230, 230, 4, 250, 7, 224, 5, 255, 235, 255, 7, 150, 5, 61, 120, 120, 70, 8, 255, 51, 255, 6, 82,
# 	143, 255, 140, 204, 255, 4, 255, 51, 7, 204, 70, 3, 0, 102, 200, 61, 230, 250, 255, 6, 51, 11, 102, 255, 255,
# 	7, 71, 255, 9, 224, 9, 7, 230, 220, 220, 220, 255, 9, 92, 112, 9, 255, 8, 255, 214, 7, 255, 224, 255, 184, 6,
# 	10, 255, 71, 255, 41, 10, 7, 255, 255, 224, 255, 8, 102, 8, 255, 255, 61, 6, 255, 194, 7, 255, 122, 8, 0, 255,
# 	20, 255, 8, 41, 255, 5, 153, 6, 51, 255, 235, 12, 255, 160, 150, 20, 0, 163, 255, 140, 140, 140, 250, 10, 15,
# 	20, 255, 0, 31, 255, 0, 255, 31, 0, 255, 224, 0, 153, 255, 0, 0, 0, 255, 255, 71, 0, 0, 235, 255, 0, 173, 255,
# 	31, 0, 255, 11, 200, 200, 255, 82, 0, 0, 255, 245, 0, 61, 255, 0, 255, 112, 0, 255, 133, 255, 0, 0, 255, 163,
# 	0, 255, 102, 0, 194, 255, 0, 0, 143, 255, 51, 255, 0, 0, 82, 255, 0, 255, 41, 0, 255, 173, 10, 0, 255, 173, 255,
# 	0, 0, 255, 153, 255, 92, 0, 255, 0, 255, 255, 0, 245, 255, 0, 102, 255, 173, 0, 255, 0, 20, 255, 184, 184, 0,
# 	31, 255, 0, 255, 61, 0, 71, 255, 255, 0, 204, 0, 255, 194, 0, 255, 82, 0, 10, 255, 0, 112, 255, 51, 0, 255, 0,
# 	194, 255, 0, 122, 255, 0, 255, 163, 255, 153, 0, 0, 255, 10, 255, 112, 0, 143, 255, 0, 82, 0, 255, 163, 255,
# 	0, 255, 235, 0, 8, 184, 170, 133, 0, 255, 0, 255, 92, 184, 0, 255, 255, 0, 31, 0, 184, 255, 0, 214, 255, 255,
# 	0, 112, 92, 255, 0, 0, 224, 255, 112, 224, 255, 70, 184, 160, 163, 0, 255, 153, 0, 255, 71, 255, 0, 255, 0,
# 	163, 255, 204, 0, 255, 0, 143, 0, 255, 235, 133, 255, 0, 255, 0, 235, 245, 0, 255, 255, 0, 122, 255, 245, 0,
# 	10, 190, 212, 214, 255, 0, 0, 204, 255, 20, 0, 255, 255, 255, 0, 0, 153, 255, 0, 41, 255, 0, 255, 204, 41, 0,
# 	255, 41, 255, 0, 173, 0, 255, 0, 245, 255, 71, 0, 255, 122, 0, 255, 0, 255, 184, 0, 92, 255, 184, 255, 0, 0,
# 	133, 255, 255, 214, 0, 25, 194, 194, 102, 255, 0, 92, 0, 255]
#
# img_path = '../../Dataset/ADE20K/Scene-Parsing/ADEChallengeData2016/annotations/validation/'
# img_save = '../../Dataset/ADE20K/Scene-Parsing/ADEChallengeData2016/annotations/color/'
# img_name = os.listdir(img_path)
#
# for name in img_name:
#     path = img_path + name
#     img = cv2.imread(path, 0)
#     out_img = Image.fromarray(img.astype('uint8'))
#     out_img.putpalette(adepallete)
#     out_img.save(img_save + name)
#
#
#
# #
# # generate boundary
# #
# import os
# import cv2
# import torch
# import os.path
# import numpy as np
#
# from torch.utils.data import Dataset
# from scipy.ndimage.morphology import distance_transform_edt
#
#
# def make_bound_dataset(split='train', data_root=None, data_list=None):
#     assert split in ['train', 'val', 'test']
#     if not os.path.isfile(data_list):
#         raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
#     image_label_list = []
#     list_read = open(data_list).readlines()
#     print("Totally {} samples in {} set.".format(len(list_read), split))
#     print("Starting Checking image&label pair {} list...".format(split))
#     for line in list_read:
#         line = line.strip()
#         line_split = line.split(' ')
#         if split == 'test':
#             if len(line_split) != 1:
#                 raise (RuntimeError("Image list file read line error : " + line + "\n"))
#             image_name = os.path.join(data_root, line_split[0])
#             label_name = image_name  # just set place holder for label_name, not for use
#             bound_name = image_name
#         else:
#             if len(line_split) != 2:
#                 raise (RuntimeError("Image list file read line error : " + line + "\n"))
#             image_name = os.path.join(data_root, line_split[0])
#             label_name = os.path.join(data_root, line_split[1])
#             bound_name = os.path.join(data_root+"/temp", line_split[1])
#
#             # VOC2012
#             # path = bound_name.split(split)[0] + split + '/' + bound_name.split(split)[1].split('/')[1] + '/'
#             # cityscapes
#             # path = bound_name.split(split)[0] + split + '/' + bound_name.split(split)[1].split('/')[1] + '/'
#             # if not os.path.exists(path):
#             #     os.makedirs(path)
#         '''
#         following check costs some time
#         if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
#             item = (image_name, label_name)
#             image_label_list.append(item)
#         else:
#             raise (RuntimeError("Image list file line error : " + line + "\n"))
#         '''
#         item = (image_name, label_name, bound_name)
#         image_label_list.append(item)
#     print("Checking image&label pair {} list done!".format(split))
#     return image_label_list
#
#
# def mask_to_onehot(mask, num_classes):
#     """
#     Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
#     hot encoding vector
#
#     """
#     _mask = [mask == i for i in range(num_classes)]
#     return np.array(_mask).astype(np.uint8)
#
#
# def onehot_to_binary_edges(mask, radius, num_classes):
#     """
#     Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)
#
#     """
#
#     if radius < 0:
#         return mask
#
#     # We need to pad the borders for boundary conditions
#     mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
#
#     edgemap = np.zeros(mask.shape[1:])
#
#     for i in range(num_classes):
#         dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
#         dist = dist[1:-1, 1:-1]
#         dist[dist > radius] = 0
#         edgemap += dist
#     edgemap = np.expand_dims(edgemap, axis=0)
#     edgemap = (edgemap > 0).astype(np.uint8)
#     return edgemap
#
#
# data_list = make_bound_dataset('train', '/home/common/XXY/Dataset/Pascal_Context',
#                                '/home/common/XXY/Project/BoundAg-CR/data/voc2010/list/train.txt')
#
# for i in range(len(data_list)):
#     image_path, label_path, bound_path = data_list[i]
#     label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
#
#     _edgemap = label
#     _edgemap = mask_to_onehot(_edgemap, 60)
#     _edgemap = onehot_to_binary_edges(_edgemap, 2, 60)
#
#     _edgemap = np.squeeze(_edgemap, axis=0)
#
#     cv2.imwrite(bound_path, _edgemap)
#     print(i, bound_path)
#


"""
compute flops and para
"""
import torch

from torch import nn
from fvcore.nn.jit_handles import get_shape, conv_flop_count
from fvcore.nn import flop_count

# from model.baseg import BASeg
from model.pspnet import PSPNet
from model.psanet import PSANet
# from model.ccnet import Seg_Model
# from model.danet import get_danet

# model = BASeg(layers=101, num_classes=150, in_channels=[256, 512, 1024, 2048], embed_dim=512,
#               multi_grid=tuple([1, 1, 1]), pretrained=False).cuda()

# model = Seg_Model(num_classes=150, criterion=None, pretrained_model=None).cuda()

# model = PSPNet(pretrained=False, num_classes=150).cuda()

model = PSANet(pretrained=False, num_classes=150).cuda()

# model = get_danet().cuda()

img = torch.randn(2, 3, 224, 224).cuda()

flops_dict, *_ = flop_count(model, img)
count = sum(flops_dict.values())
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"FLOPs: {count:.3f}G\n"
      f"#param: {n_parameters / 1e6:.3f}M")
