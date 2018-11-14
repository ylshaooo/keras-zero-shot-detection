import os

import numpy as np


def box_iou(b1, b2):
    """
    :param b1: shape=(i, 4)
    :param b2: shape=(j, 4)
    :return: iou: shape=(i, j)
    """
    b1 = np.expand_dims(b1, -2)
    b1_mins = b1[..., :2]
    b1_maxes = b1[..., 2:4]
    b1_wh = b1_maxes - b1_mins

    b2 = np.expand_dims(b2, 0)
    b2_mins = b2[..., :2]
    b2_maxes = b2[..., 2:4]
    b2_wh = b2_maxes - b2_mins

    intersect_mins = np.maximum(b1_mins, b2_mins)
    intersect_maxes = np.minimum(b1_maxes, b2_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)
    return iou


def get_box_info(file):
    with open(file) as f:
        lines = f.readlines()
        classes = []
        boxes = []
        for line in lines:
            box_info = line.strip().split(' ')
            classes.append(unseen_classes.index(box_info[0]))
            for i in box_info[-4:]:
                boxes.append(int(i))

    boxes = np.reshape(boxes, (-1, 4))
    return classes, boxes


unseen_classes = ['car', 'dog', 'sofa', 'train']
result_matrix = np.zeros((4, 4), dtype='int32')

prediction = os.listdir('data/predicted/test')

for p in prediction:
    pred_classes, pred_boxes = get_box_info(os.path.join('data/predicted/test', p))
    gt_classes, gt_boxes = get_box_info(os.path.join('data/ground-truth/test', p))
    if len(pred_classes) == 0 or len(gt_classes) == 0:
        continue
    ious = box_iou(pred_boxes, gt_boxes)
    for i in range(len(ious)):
        if ious[i].max() < 0.5:
            continue
        gt_class = gt_classes[np.argmax(ious[i])]
        pred_class = pred_classes[i]
        result_matrix[gt_class, pred_classes] += 1

print(result_matrix)
