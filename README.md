# SSD_MobileNet
SSD: Single Shot MultiBox Detector | a PyTorch Model for Object Detection | VOC , COCO | Custom Object Detection  

This repo contains code for [Single Shot Multibox Detector (SSD)](https://arxiv.org/abs/1512.02325) with custom backbone networks. The authors' original implementation can be found [here](https://github.com/weiliu89/caffe/tree/ssd).


#### Objects' Bounding Boxes

For each image, the bounding boxes of the ground truth objects follows (x_min, y_min, x_max, y_max) format`.

# Training
* In config.json change the paths. 
* "backbone_network" : "MobileNetV2" or "MobileNetV1"
* For training run
  ```
  python train.py config.json
  ```
# Inference 
  ```
  python inference.py image_path checkpoint
  ```
 
