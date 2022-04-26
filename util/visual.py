import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from model.baseg import BASeg

input_data = [] 
output_data = []
def get_gradient(model, inputs, outputs):
    input_data.append(inputs)
    output_data.append(outputs)

activation={}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def compute_dpixel(src_x, src_w, dst_w):
    scale_x = src_w / dst_w
    shift = 0.5 * (scale_x - 1)
    dst_x = (src_x - shift) * dst_w / src_w

    return dst_x

def compute_rpixel(dst_x, src_w, dst_w):
    scale_x = src_w / dst_w
    shift = 0.5 * (scale_x - 1)
    src_x = scale_x * dst_x + shift

    return src_x

# model_path = './exp/cityscapes/attenet101/model/baseline_aspp_canny_gate_lre2_flohem_tv/train_epoch_200.pth'
# image_path = '../../Dataset/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_002196_leftImg8bit.png'

model_path = './exp/ade20k/attenet101/model/baseline_aspp_canny_gate_lr5e3/train_epoch_150.pth'
image_path = '../../Dataset/ADE20K/Scene-Parsing/ADEChallengeData2016/images/validation/ADE_val_00000174.jpg'

model = AtteNet(layers=101, num_classes=19, in_channels=[256, 512, 1024, 2048], embed_dim=512,
                depth=[1], multi_grid=tuple([1, 1, 1]), pretrained=True) 

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model = model.to(torch.device('cuda:0'))

transform = transforms.Compose([
    transforms.Resize([520, 520]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open(image_path)
img = transform(img).unsqueeze(0).cuda()

model.eval()
model.cont_aggreg.register_forward_hook(get_gradient)
model.layer1.register_forward_hook(get_activation('Context_Aggregation'))
result = model(img)

cont_aggreg_1 = output_data[0]
cont_aggreg_2 = activation['Context_Aggregation']

print(compute_rpixel(350, cont_aggreg_2.shape[-1], result[0].shape[-1]))
print(compute_rpixel(500, cont_aggreg_2.shape[-1], result[0].shape[-1]))
print(compute_dpixel(220, cont_aggreg_2.shape[-1], result[0].shape[-1]))
print(compute_dpixel(250, cont_aggreg_2.shape[-1], result[0].shape[-1]))

point = cont_aggreg_2
point = point[:, :, 100, 125].unsqueeze(2).unsqueeze(3)
point = point.expand(point.size())

visual = torch.cosine_similarity(point, cont_aggreg_2, dim=1)
visual = visual.unsqueeze(1)
visual = F.interpolate(visual, size=result[0].size()[2:], mode='bilinear', align_corners=True)

plt.figure(figsize=(12, 12))
plt.imshow(visual.cpu().detach().numpy()[0, 0, :, :], cmap='jet')
plt.axis('off')
plt.savefig('./exp/cityscapes/baseg101/result/baseline_aspp_canny_gate_lre2_flohem_tv/ms/validation/visual/point' + '.png', dpi=100, bbox_inches='tight')

# for i in range(10):
#     plt.imshow(cont_aggreg_1.cpu().detach().numpy()[0, i, :, :], cmap='jet')
#     plt.axis('off')
#     plt.savefig('./exp/cityscapes/baseg101/result/baseline_aspp_canny_gate_lre2_flohem_tv/ms/validation/visual/visual1_'+ str(i) + '.png', dpi=100, bbox_inches='tight')

# for i in range(10):
#     plt.imshow(cont_aggreg_2.cpu().detach().numpy()[0, i, :, :], cmap='jet')
#     plt.axis('off')
#     plt.savefig('./exp/cityscapes/baseg101/result/baseline_aspp_canny_gate_lre2_flohem_tv/ms/validation/visual/visual2_'+ str(i) + '.png', dpi=100, bbox_inches='tight')

