"""
Train the YOLO model for your own dataset.
"""

import os

import keras
import keras.backend as K
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam

from yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss
from yolo3.utils import get_random_data, normalize


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, embedding, anchors, input_shape, batch_size=8, shuffle=True):
        self.data = data
        self.indexes = np.arange(len(self.data))
        self.embedding = embedding
        self.anchors = anchors
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, index):
        batch_index = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self.data[k] for k in batch_index]
        x, y = self.data_generator(batch_data)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generator(self, batch_data):
        num_classes = self.embedding.shape[0]
        embedding = np.tile(np.expand_dims(self.embedding, 0), (self.batch_size, 1, 1))

        image_data = []
        box_data = []
        for data in batch_data:
            image, box = get_random_data(data, self.input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, self.input_shape, self.anchors, num_classes)
        return [image_data, *y_true, embedding], np.zeros(self.batch_size)


def _main():
    annotation_path = 'data/train.txt'
    embedding_path = 'data/glove_embedding.npy'
    log_dir = 'logs/voc2012/'
    anchors_path = 'model_data/yolo_anchors.txt'
    weights_path = 'model_data/yolo_weights.h5'
    anchors = get_anchors(anchors_path)
    num_seen = 16

    input_shape = (416, 416)  # multiple of 32, hw
    embedding_shape = (20, 300)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    batch_size = 8
    embedding = np.load(embedding_path)
    embedding = normalize(embedding)

    model = create_model(input_shape, embedding_shape, anchors, num_seen, weights_path=weights_path)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(DataGenerator(lines[:num_train], embedding, anchors, input_shape, batch_size),
                        validation_data=DataGenerator(lines[num_train:], embedding, anchors, input_shape, batch_size),
                        epochs=10,
                        initial_epoch=0,
                        workers=2,
                        use_multiprocessing=True,
                        callbacks=[logging, checkpoint])

    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    print('Unfreeze all of the layers.')

    model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(DataGenerator(lines[:num_train], embedding, anchors, input_shape, batch_size),
                        validation_data=DataGenerator(lines[num_train:], embedding, anchors, input_shape, batch_size),
                        epochs=80,
                        initial_epoch=10,
                        workers=2,
                        use_multiprocessing=True,
                        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    print('Finish training and save weights.')
    model.save_weights(log_dir + 'trained_weights_sigmoid.h5')
    K.clear_session()


def get_classes(classes_path):
    """loads the classes"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, embedding_shape, anchors, num_seen,
                 load_pretrained=True, weights_path='model_data/yolo_weights.h5'):
    """create the training model"""
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    num_classes, _ = embedding_shape

    y_true = [Input(shape=(h // [32, 16, 8][l], w // [32, 16, 8][l], num_anchors // 3,
                           5 + num_classes)) for l in range(3)]
    y_embedding = Input(shape=embedding_shape)

    model_body = yolo_body(image_input, num_anchors // 3)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        num = len(model_body.layers) - 3
        for i in range(num):
            model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(
        yolo_loss,
        output_shape=(1,),
        name='yolo_loss',
        arguments={'anchors': anchors, 'num_seen': num_seen, 'ignore_thresh': 0.5}
    )([*model_body.output, *y_true, y_embedding])
    model = Model([model_body.input, *y_true, y_embedding], model_loss)

    return model


if __name__ == '__main__':
    _main()
