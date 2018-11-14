import keras.backend as K
import numpy as np
from PIL import Image
from keras.layers import Input
from keras.models import Model
from tqdm import tqdm

from .darknet import darknet_body
from .utils import letterbox_image

sess = K.get_session()

input_image = Input(shape=(416, 416, 3))
model = Model(input_image, darknet_body(input_image))
model.load_weights('weights/yolo_weights.h5', by_name=True, skip_mismatch=True)

feature = {'car': [], 'dog': [], 'sofa': [], 'train': []}
with open('data/location.txt') as f:
    lines = f.readlines()

for line in tqdm(lines):
    info = line.strip().split(' ')
    img = Image.open(info[0])
    w, h = img.size
    img = letterbox_image(img, (416, 416))
    img_data = np.array(img, dtype='float32') / 255.
    img_data = np.expand_dims(img_data, 0)
    feature_map = sess.run(model.output, feed_dict={model.input: img_data})

    obj_location = info[1:]
    for i in range(0, len(obj_location), 3):
        # mapping the object location into the darknet feature map
        c = obj_location[i]
        x = int(int(obj_location[i + 1]) * 13 / w)
        y = int(int(obj_location[i + 2]) * 13 / h)
        feature[obj_location[i]].append(feature_map[0, y, x, :])

K.clear_session()

for c in feature.keys():
    # feature save as npy files
    feature[c] = np.squeeze(feature[c])
    np.save('data/yolo_feat_{}.npy'.format(c), feature[c])
