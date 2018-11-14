"""Get bounding box to crop objects."""

import xml.etree.ElementTree as ET


def convert_annotation(image_path, file):
    image_name = image_path.split('/')[-1]
    in_file = open('data/VOCdevkit/VOC2012/Annotations/%s' % image_name.replace('jpg', 'xml'))
    tree = ET.parse(in_file)
    root = tree.getroot()

    file.write(image_path)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if int(difficult) == 1:
            continue
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
             int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        file.write(' {} {} {} {} {}'.format(cls, b[0], b[1], b[2], b[3]))
    file.write('\n')


if __name__ == '__main__':
    with open('data/image.txt') as f:
        lines = f.readlines()
    with open('data/box.txt', 'w') as f:
        for line in lines:
            convert_annotation(line.strip(), f)
