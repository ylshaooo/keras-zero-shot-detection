"""
Generate ground truth file for each test image of VOC2012 dataset.
"""

import xml.etree.ElementTree as ET
import os

seen_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'cat', 'chair', 'cow', 'diningtable',
                'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'tvmonitor']

unseen_classes = ['car', 'dog', 'sofa', 'train']

total_classes = seen_classes + unseen_classes


def convert_annotation(image_id):
    in_file = open('data/voc/VOCdevkit/VOC2012/Annotations/%s.xml' % image_id)
    tree = ET.parse(in_file)
    root = tree.getroot()

    with open('data/voc/ground-truth/test/%s.txt' % image_id, 'w') as f:
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if int(difficult) == 1:
                continue
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
                 int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            f.write('{} {} {} {} {}\n'.format(cls, b[0], b[1], b[2], b[3]))


if __name__ == '__main__':
    with open('data/voc/test.txt') as f:
        lines = f.readlines()
    for line in lines:
        img_id = line.split('/')[-1].split('.')[0]
        convert_annotation(img_id)
