import os
import cv2
import numpy as np

from PIL import Image, ImageChops

predi_pa = './exp/cityscapes/attenet101/result/baseline_aspp_canny_gate_lre2_flohem_tv/ms/validation/visual/'
predi_pa = '../../Project/mmsegmentation/show_dirs/psanet_r101_d8_769x769_80k_cityscapes/frankfurt/'
truth_pa = '../../Dataset/cityscapes/gtFine/val/frankfurt/'

name_pred = 'frankfurt_000001_003056_leftImg8bit.png'
name_trut = 'frankfurt_000001_003056_gtFine_color.png'

truth_img = cv2.imread(truth_pa+name_trut).astype('uint8')
predi_img = cv2.imread(predi_pa+name_pred).astype('uint8')
predi_img[np.where(truth_img == 0)] = 0

diff = cv2.absdiff(predi_img, truth_img)
diff = 255 - diff

cv2.imwrite('./exp/cityscapes/attenet101/result/baseline_aspp_canny_gate_lre2_flohem_tv/ms/validation/visual/error.png', diff)
cv2.imshow('diff', diff)
cv2.waitKey(1000000)