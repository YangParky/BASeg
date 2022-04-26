import os
import cv2
import torch
import os.path
import numpy as np

from torch.utils.data import Dataset
from scipy.ndimage.morphology import distance_transform_edt


def make_bound_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
            bound_name = image_name
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
            bound_name = os.path.join(data_root+"/temp", line_split[1])

            # VOC2012
            # path = bound_name.split(split)[0] + split + '/' + bound_name.split(split)[1].split('/')[1] + '/'
            # cityscapes
            # path = bound_name.split(split)[0] + split + '/' + bound_name.split(split)[1].split('/')[1] + '/'
            # if not os.path.exists(path):
            #     os.makedirs(path)
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name, bound_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)


def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)

    """

    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap


data_list = make_bound_dataset('train', '/home/common/XXY/Dataset/Pascal_Context', 
                               '/home/common/XXY/Project/BoundAg-CR/data/voc2010/list/train.txt')

for i in range(len(data_list)):
    image_path, label_path, bound_path = data_list[i]
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

    _edgemap = label
    _edgemap = mask_to_onehot(_edgemap, 60)
    _edgemap = onehot_to_binary_edges(_edgemap, 2, 60)

    _edgemap = np.squeeze(_edgemap, axis=0)

    cv2.imwrite(bound_path, _edgemap)
    print(i, bound_path)
