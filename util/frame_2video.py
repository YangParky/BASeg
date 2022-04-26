import os
import cv2

fps = 60
fourcc = cv2.VideoWriter_fourcc('m','p','4','v') 

dat_path = './data/cityscapes/list/demoVideo.txt' 
img_path = './exp/cityscapes/baseg101/video/baseline_aspp_canny_gate_lre3_floss/ss/color/'
vid_path = './exp/cityscapes/baseg101/video/baseline_aspp_canny_gate_lre3_floss/ss/convert/color2.avi'

videoWriter = cv2.VideoWriter(vid_path, fourcc, fps, (2048, 1024))
img_name = []

with open(dat_path, 'r') as f:
	lines = f.readlines()
	for line in lines:
		line = line.strip()
		line = line.split('/')[-1]
		img_name.append(line)

for i in range(len(img_name)):

    img = cv2.imread(img_path+img_name[i])
    videoWriter.write(img)

videoWriter.release()
print('finished~')