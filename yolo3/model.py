"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4),
                           'padding': 'valid' if kwargs.get('strides') == (2, 2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(DarknetConv2D(*args, **no_bias_kwargs),
                   BatchNormalization(),
                   LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    """A series of resblocks starting with a downsampling Convolution2D"""
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
                    DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    """Darknet body having 52 Convolution2D layers"""
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    """6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer"""
    x = compose(DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
                DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
                DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
                DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
                DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
                DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def yolo_body(inputs, num_seen, num_anchors):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (4 + num_seen) + 64)

    x = compose(DarknetConv2D_BN_Leaky(256, (1, 1)),
                UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors * (4 + num_seen) + 64)

    x = compose(DarknetConv2D_BN_Leaky(128, (1, 1)),
                UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors * (4 + num_seen) + 64)

    return Model(inputs, [y1, y2, y3])


def yolo_head(feats, anchors, num_seen, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters.

    xy - relative to grid shape
    wh - relative to input shape
    """
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    box_attribute = K.reshape(feats[..., :64], [-1, grid_shape[0], grid_shape[1], 1, 64])
    feats = K.reshape(feats[..., 64:], [-1, grid_shape[0], grid_shape[1], num_anchors, 4 + num_seen])

    # Adjust predictions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    obj_prob = K.sigmoid(feats[..., 4:])
    box_attribute = K.sigmoid(box_attribute)

    if calc_loss:
        return grid, feats, box_xy, box_wh, box_attribute
    return box_xy, box_wh, box_attribute, obj_prob


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """Get corrected boxes"""
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, attributes, num_seen, num_classes,
                          input_shape, image_shape):
    """Process Conv layer output"""
    box_xy, box_wh, box_attribute, object_prob = yolo_head(feats, anchors, num_seen, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])

    for _ in range(4):
        attributes = K.expand_dims(attributes, 0)
    box_confidence = K.max(object_prob, axis=-1, keepdims=True)

    num_unseen = num_classes - num_seen
    box_class_probs = cosine_similarity(K.expand_dims(box_attribute, -2), attributes)
    box_class_probs = K.one_hot(K.argmax(box_class_probs[..., num_seen:], -1), num_unseen)
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_unseen])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_seen,
              attribute,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    num_classes, _ = attribute.shape
    attribute = K.cast(attribute, K.dtype(yolo_outputs[0]))

    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]],
                                                    attribute, num_seen, num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, 0)
    box_scores = K.concatenate(box_scores, 0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes - num_seen):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold
        )
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, 0)
    scores_ = K.concatenate(scores_, 0)
    classes_ = K.concatenate(classes_, 0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(num_anchors, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape=(m, h, w, 3, 5 + num_seen) like yolo_outputs, xywh are relative value

    """
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than seen classes'
    num_layers = len(anchors) // 3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // [32, 16, 8][l] for l in range(num_layers)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


def box_iou(b1, b2):
    """Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    """

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def cosine_similarity(tensor0, tensor1, axis=-1):
    """Calculate cosine similarity between two attribute vectors"""
    tensor0_norm = K.sqrt(K.sum(K.square(tensor0), axis=axis))
    tensor1_norm = K.sqrt(K.sum(K.square(tensor1), axis=axis))
    inner_prod = K.sum(tensor0 * tensor1, axis=axis) / (tensor0_norm * tensor1_norm)
    return inner_prod


def hinge_loss(y_true, y_pred, true_class_index, num_seen):
    """Calculate max margin loss of predicted attributes

    Parameters
    ----------
    y_true: attribute matrix, shape=(b, 1, 1, 1, num_seen, 64)
    y_pred: yolo output attributes, shape=(b, h, w, anchors, 64)
    true_class_index: class index of ground truth attribute, shape=(b, h, w, anchors, num_seen)
    num_seen: number of seen classes
    """
    y_pred = K.expand_dims(y_pred, -2)
    scores = cosine_similarity(y_pred, y_true)  # shape=(b, h, w, anchors, num_seen)
    true_class_scores = K.max(true_class_index * scores, -1)  # shape=(b, h, w, anchors)
    loss = 0
    for i in range(num_seen):
        loss += K.maximum(0., 0.2 - true_class_scores + scores[..., i])
    loss = K.expand_dims(loss - 1, -1)
    return loss


def cross_entropy_loss(y_true, y_pred, true_class_index):
    """Calculate loss of predicted attributes in embarrassing algorithm

    Parameters
    ----------
    y_true: attribute matrix, shape=(b, 1, 1, 1, num_seen, 64)
    y_pred: yolo output attributes, shape=(b, h, w, anchors, 64)
    true_class_index: class index of ground truth attribute, shape=(b, h, w, anchors, num_seen)
    """
    y_pred = K.expand_dims(y_pred, -2)
    pred_class = K.softmax(cosine_similarity(y_true, y_pred), -1)
    loss = K.categorical_crossentropy(true_class_index, pred_class)
    loss = K.expand_dims(loss, -1)
    return loss


def class_relation(true_class_index, attribute):
    """
    Parameters
    ----------
    true_class_index: class index of ground truth attribute, shape=(b, h, w, anchors, num_seen)
    attribute: attribute matrix, shape=(b, 1, 1, 1, num_seen, 64)
    """
    true_class_index = K.expand_dims(true_class_index, -1)
    true_class_attribute = K.max(true_class_index * attribute, axis=-2, keepdims=True)
    relation = cosine_similarity(true_class_attribute, attribute)
    return K.relu(relation)


def yolo_loss(args, anchors, num_seen, ignore_thresh=.5):
    """Return yolo_loss tensor

    Parameters
    ----------
    args: [*yolo_outputs, *y_true, y_attribute]
    # yolo_outputs: list of tensor, the output of yolo_body
    # y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_seen: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss
    Returns
    -------
    loss: tensor, shape=(1,)

    """
    num_layers = len(anchors) // 3  # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:-1]  # shape=(num_layers, b, h, w, anchors, 5 + num_seen)
    attributes = args[-1]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0.
    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))
    for _ in range(3):
        attributes = K.expand_dims(attributes, 1)

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh, pred_box_attribute = yolo_head(
            yolo_outputs[l], anchors[anchor_mask[l]], num_seen, input_shape, calc_loss=True
        )
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            mask = mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, mask

        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *arg: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        raw_pred_xy = raw_pred[..., 0:2]
        raw_pred_wh = raw_pred[..., 2:4]
        raw_pred_objectness = raw_pred[..., 4:]
        raw_pred_box_attribute = pred_box_attribute
        true_relation = class_relation(true_class_probs, attributes)

        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred_xy, True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred_wh)
        object_loss = object_mask * K.binary_crossentropy(object_mask * true_relation, raw_pred_objectness, True) + \
                      (1 - object_mask) * \
                      K.binary_crossentropy(object_mask * true_relation, raw_pred_objectness, True) * ignore_mask
        attribute_loss = object_mask * hinge_loss(attributes, raw_pred_box_attribute, true_class_probs, num_seen)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        object_loss = K.sum(object_loss) / mf
        attribute_loss = K.sum(attribute_loss) / mf
        loss += xy_loss + wh_loss + object_loss + attribute_loss
    return loss
