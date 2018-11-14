import random

import numpy as np
from PIL import Image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from tqdm import tqdm


model = ResNet50(include_top=False, weights='weights/resnet_weights.h5', pooling='avg')

feature = {'car': [], 'dog': [], 'sofa': [], 'train': []}
with open('data/box.txt') as f:
    lines = f.readlines()

cnt = 0
for line in tqdm(lines):
    info = line.strip().split(' ')
    img = Image.open(info[0])

    for i in range(1, len(info), 5):
        # crop object from img
        box = img.crop((int(info[i + 1]), int(info[i + 2]), int(info[i + 3]), int(info[i + 4])))

        # Warp object to a fixed size.
        # Simulate roi pooling layer in Faster RCNN.
        box = box.resize((224, 224))
        if random.random() > 0.96:
            # randomly save 4% of the warped image to see the effect of warping
            box.save('images/random/{:0>5d}_{}.jpg'.format(cnt, info[i]))
            cnt += 1
        box_data = image.img_to_array(box)
        box_data = np.expand_dims(box_data, 0)
        box_data = preprocess_input(box_data)
        feature[info[i]].append(model.predict(box_data))

for c in feature.keys():
    # feature save as npy files
    feature[c] = np.squeeze(feature[c])
    np.save('data/resnet_feat_{}.npy'.format(c), feature[c])
