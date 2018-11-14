"""
Generate ground truth file for each test image of VOC2012 dataset.
"""

import xml.etree.ElementTree as ET

# change your own spilt in voc_classes.txt
num_seen = 16
with open('data/voc_classes.txt') as f:
    classes = f.readlines()

total_classes = [c.strip() for c in classes]
seen_classes = total_classes[:num_seen]
unseen_classes = total_classes[num_seen:]


def convert_annotation(image_id):
    """Convert annotations from xml files to txt files as mAP program required."""

    in_file = open('data/voc/VOCdevkit/VOC2012/Annotations/%s.xml' % image_id)
    tree = ET.parse(in_file)
    root = tree.getroot()

    with open('data/voc/ground-truth/test/%s.txt' % image_id, 'w') as f:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
                 int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            f.write('{} {} {} {} {}\n'.format(cls, b[0], b[1], b[2], b[3]))


if __name__ == '__main__':
    with open('data/test.txt') as f:
        lines = f.readlines()
    for line in lines:
        img_id = line.split('/')[-1].split('.')[0]
        convert_annotation(img_id)
