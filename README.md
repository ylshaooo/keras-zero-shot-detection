# keras-Zero-Shot-Detection

## Introduction

A Keras implementation of Zero-Shot Detection Model based on YOLOv3 (Tensorflow backend),
referring to [keras-yolo3](https://github.com/qqwweee/keras-yolo3).

- ### Object Detection

    Object detection is a computer vision task that deals with detecting instances of
    semantic objects of a certain class (such as humans, buildings or cars) in digital
    images and videos, if present, to return the spatial location and extent of each
    object instance (e.g., via a bounding box). 

- ### Zero-Shot Learning

    Zero-shot learning (ZSL) aims to minimize the annotation requirements by enabling
    recognition of unseen classes, i.e. those with no training examples. This is achieved
    by transferring knowledge from seen to unseen classes by means of auxiliary data,
    typically obtained easily from textual sources.

- ### Zero-Shot Detection

    The existing ZSL approaches predominantly focus on classification problems. While
    zero-shot object detection (ZSD) task aims to recognize and localize instances of
    object classes with no training examples, purely based on auxiliary information that
    describes the class characteristics. We apply the mainstream ZSL approach on the
    famous [YOLO](https://arxiv.org/abs/1506.02640) detection model to achieve this goal.
    
    

---

## Results on PASCAL VOC

- ### Dataset Split

    We split **16 / 4** of total 20 VOC classes as seen/unseen classes. We guarantee that
    there must be at least one category in seen classes that are semantically similar to
    any unseen classes. Unseen classes are as follows.

- ### MAP of Unseen Classes
    
    U2U detection results for VOC dataset:
    
    |   car  |   dog  |  sofa  | train  |   mAP  |
    |:------:|:------:|:------:|:------:|:------:|
    | 12.84% | 93.48% | 55.93% | 53.28% | 53.88% |

- ### Feature Exaction

    For further research,

---

## Train and Evaluate

1. Generate your own annotation file and class names file.  
    One row for one image.  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`.  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2.  Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/). The
file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training `python train.py`. Use your trained weights or
checkpoint weights in yolo.py. Remember to modify class path or anchor path.

4. Test ZSD model and evaluate the results using [MAP](https://github.com/Cartucho/mAP), 
or run the visualization demo. `python test.py  OR  python demo.py`.  
    Test file are in the form: `path/to/img`, one row for one image.  

---

## Issues to know

1. The train and test environment is
    - Python 3.5.5
    - Keras 2.2.0
    - tensorflow 1.6.0

2. Default yolo anchors are used. If you use your own anchors, probably some changes are
needed.

3. The inference result is not totally the same as Darknet but the difference is small.

4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

5. Always load pretrained weights and freeze layers in the first stage of training. Or
try Darknet training. It's OK if there is a mismatch warning.

6. The training strategy is for reference only. Adjust it according to your dataset and
your goal. And add further strategy if needed.
