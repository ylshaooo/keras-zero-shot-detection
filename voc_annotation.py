import os
import xml.etree.ElementTree as ET


# change your own spilt in voc_classes.txt
with open('data/voc_classes.txt') as f:
    classes = f.readlines()
classes = [c.strip() for c in classes]


def convert_annotation(image_id, list_file):
    """convert annotations from xml files to txt files"""

    in_file = open('data/VOCdevkit/VOC2012/Annotations/%s.xml' % image_id)
    tree = ET.parse(in_file)
    root = tree.getroot()

    list_file.write('data/VOCdevkit/VOC2012/JPEGImages/%s.jpg' % image_id)
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        # in the form (x1, y1, x2, y2)
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        list_file.write(' ' + ','.join([str(a) for a in b]) + ',' + str(cls_id))
    list_file.write('\n')


if __name__ == '__main__':
    xml_files = os.listdir('data/VOCdevkit/VOC2012/Annotations')
    train_file = open('data/train.txt', 'w')
    for xml_file in xml_files:
        img_id = xml_file.split('.')[0]
        convert_annotation(img_id, train_file)
    train_file.close()
