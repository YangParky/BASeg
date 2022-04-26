import numpy as np
from multiprocessing import Pool
from math import floor


def eval_mask_boundary(seg_mask, gt_mask, num_classes, num_proc=12, bound_th=0.0003):

    p = Pool(processes=num_proc)
    seg_mask = np.expand_dims(seg_mask,0)
    gt_mask = np.expand_dims(gt_mask, 0)
    batch_size = seg_mask.shape[0]

    Fpc = np.zeros(num_classes)
    Fc = np.zeros(num_classes)
    # mIoU = np.zeros(num_classes)

    for class_id in range(num_classes):
        args = [((seg_mask[i] == class_id).astype(np.uint8),
                 (gt_mask[i] == class_id).astype(np.uint8),
                 gt_mask[i] == 255,
                 bound_th)
                for i in range(batch_size)]
        temp = p.map(db_eval_boundary_wrapper, args)
        # foreground_mask, gt_mask, ignore, bound_th = args
        # temp = db_eval_boundary(foreground_mask, gt_mask, ignore, bound_th)
        temp = np.array(temp)
        Fs = temp[:, 0]
        _valid = ~np.isnan(Fs)
        Fc[class_id] = np.sum(_valid)
        Fs[np.isnan(Fs)] = 0
        Fpc[class_id] = sum(Fs)
        # mIoU[class_id] = sum(temp[:, -1])

    p.close()
    
    return Fpc, Fc


def db_eval_boundary_wrapper(args):
    foreground_mask, gt_mask, ignore, bound_th = args
    return db_eval_boundary(foreground_mask, gt_mask, ignore, bound_th)


def db_eval_boundary(foreground_mask, gt_mask, ignore_mask, bound_th=0.008):
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    foreground_mask[ignore_mask] = 0
    gt_mask[ignore_mask] = 0

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask);
    gt_boundary = seg2bmap(gt_mask);

    from skimage.morphology import binary_dilation, disk

    fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary, disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
        # mIoU = np.sum(fg_match) / float(n_gt)

    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
        # mIoU = np.sum(gt_match) / float(n_fg)

    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
        # mIoU = 1

    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)
        # mIoU = (np.sum(fg_match) + np.sum(gt_match)) / (float(n_fg) + float(n_gt))

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall);

    return F, precision


def seg2bmap(seg, width=None, height=None):
    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) > 0.01), \
        'Can''t convert %dx%d seg to %dx%d bmap.' % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + floor((y - 1) + height / h)
                    i = 1 + floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


if __name__ == "__main__":
    import cv2
    import os

    target_path = '../../Dataset/cityscapes/'
    pred_path = './exp/cityscapes/baseg101/result/baseline_aspp_canny_gate_lre3/ms/validation/gray/'
    target_names = []
    figure_names = []

    with open('./data/cityscapes/list/fine_val.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split(' ')

            figure_names.append(line[0].split('/')[3])
            target_names.append(target_path + line[1])
    
    f_score = 0
    f_percl = 0
    # mIoU = 0
    for i in range(len(target_names)):
        target = cv2.imread(target_names[i], cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path + figure_names[i], cv2.IMREAD_GRAYSCALE)
        
        fpc,fc = eval_mask_boundary(pred,target,19)
        f_multiple = fpc/fc
        f_percl += f_multiple
        f_score += np.sum(f_multiple)/19
        # mIoU += np.sum(m)/19

    f_score = f_score / len(target_names)
    f_percl = f_percl / len(target_names)
    # mIoU = mIoU / len(target_names)
    
    print('f_score', str(f_score))
    print('f_percl', str(f_percl))
    # print('mIoU', str(mIoU))
