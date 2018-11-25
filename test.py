"""
Test zero-shot YOLO detection model on unseen classes.
"""

import os

import numpy as np
from PIL import Image
import keras.backend as K
import keras.layers as KL
from sklearn.preprocessing import normalize
from tqdm import tqdm

from yolo3.model import yolo_body, yolo_plus_body, yolo_eval
from yolo3.utils import letterbox_image

seen_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'cat', 'chair', 'cow', 'diningtable',
                'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'tvmonitor']

unseen_classes = ['car', 'dog', 'sofa', 'train']

total_classes = seen_classes + unseen_classes


class YOLO(object):
    def __init__(self):
        self.weight_path = 'logs/voc/trained_weights_final.h5'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.embedding_path = 'model_data/glove_embedding.npy'
        self.predict_dir = 'data/predicted/'
        self.score = 0.1
        self.iou = 0.5
        self.num_seen = 16
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        weight_path = os.path.expanduser(self.weight_path)
        assert weight_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        image_inut = KL.Input(shape=self.model_image_size + (3,))
        anchor_input = KL.Input(shape=(num_anchors, 2))
        self.yolo_model = yolo_body(image_inut, num_anchors // 3)
        self.yolo_plus_model = yolo_plus_body([image_inut, anchor_input], self.yolo_model.outputs, num_anchors // 3)
        self.yolo_plus_model.load_weights(self.weight_path)
        print('{} model, anchors and classes loaded.'.format(weight_path))

        # Generate output tensor targets for filtered bounding boxes.
        embeddings = np.load(self.embedding_path)
        embeddings = normalize(embeddings)
        boxes, scores, classes = yolo_eval(self.yolo_plus_model.output, self.anchors, self.num_seen,
                                           embeddings, self.model_image_size,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image_path):
        image = Image.open(image_path)

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        anchors = np.expand_dims(self.anchors, 0)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_plus_model.inputs[0]: image_data,
                self.yolo_plus_model.inputs[1]: anchors,
                K.learning_phase(): 0
            }
        )

        image_name = image_path.split('/')[-1].split('.')[0]
        with open(os.path.join(self.predict_dir, image_name + '.txt'), 'w') as f:
            for i, c in enumerate(out_classes):
                class_name = unseen_classes[c]
                confidence = out_scores[i]
                box = out_boxes[i]

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                f.write('{} {} {} {} {} {}\n'.format(class_name, confidence, left, top, right, bottom))

    def close_session(self):
        self.sess.close()


def _main():
    test_path = 'data/test.txt'

    yolo = YOLO()
    with open(test_path) as rf:
        test_img = rf.readlines()
    test_img = [c.strip() for c in test_img]

    K.clear_session()   # get a new session
    for img_path in tqdm(test_img):
        img_path = img_path.split()[0]
        yolo.detect_image(img_path)
    K.clear_session()


if __name__ == '__main__':
    _main()
