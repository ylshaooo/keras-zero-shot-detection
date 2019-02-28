import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from keras import backend as K
from keras.layers import Input

from yolo3.model import yolo_body, yolo_eval
from yolo3.utils import letterbox_image

unseen_classes = ['car', 'dog', 'horse', 'sofa', 'train']


class YOLO(object):
    def __init__(self):
        self.weight_path = 'logs/voc/trained_weights_final.h5'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.attribute_path = 'model_data/attributes.npy'
        self.score = 0.4
        self.iou = 0.5
        self.num_seen = 16
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

        hsv_tuples = [(x / len(unseen_classes), 1., 1.) for x in range(len(unseen_classes))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.weight_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)

        self.yolo_model = yolo_body(Input(shape=(None, None, 3)), self.num_seen, num_anchors // 3)
        self.yolo_model.load_weights(self.weight_path, by_name=True)
        print('{} model, anchors and classes loaded.'.format(model_path))

        # Generate output tensor targets for filtered bounding boxes.
        attribute = np.load(self.attribute_path)
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors, self.num_seen,
                                           attribute, self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(5e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 250

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = unseen_classes[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()


def detect_img(yolo):
    while True:
        img_path = input('Input image filename: ')
        if not img_path:
            break
        try:
            image = Image.open(img_path)
        except OSError as e:
            print(e)
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()


if __name__ == '__main__':
    detect_img(YOLO())
