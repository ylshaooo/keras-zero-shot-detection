# Feature Extraction and Visualization

Use T-SNE tools to cluster and visualize features: object feature map extracted by ResNet50 
in Faster RCNN and the grid feature used in YOLO.

---

### Faster RCNN Feature

1. Run `get_box.py` to get object bounding boxes from annotations.
2. Run `resnet_feature.py` to crop images and extract ResNet50 features of class objects.

### YOLO Feature

1. Run `get_loc.py` to get object location from annotations.
2. Run `darknet_feature.py` to locate and extract grid features by DarkNet.

---

Run `visualize.py` to cluster and visualize feature distribution.
