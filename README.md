
# MMDetection-Plus

## Introduction

The master branch works with **PyTorch 1.1** or higher.

mmdetection-plus is a enhanced object detection codebase for my research and projects based on [mmdetection](https://github.com/open-mmlab/mmdetection.git), which is developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

![demo image](demo/coco_test_12510.jpg)


## Modification
* Modify [mmdet/apis/inference.py](mmdet/apis/inference.py)
  * In line 143:
    ```python
    show=out_file is None,
    out_file=out_file
    ```
* Add [inference code](tools/infer.py) for easily building detection web service
* Add VOC-COCO, YOLO-VOC data converter


## How to use mmdetection-plus in your projects?
1. [Install mmdetection](INSTALL.md).
2. Prepare data.
3. Modify config py:
  > bbox_head: num_classes = cls_num + 1  
  > total_epoch
4. Modify mmdet/core/evaluation/class_names.py
5. Modify mmdet/datasets/voc.py:  
  > CLASSES = ('xx', 'xxx')

**Note: If you only have 1 class, use CLASSES = [‘xx’], or you will get some trouble.**

6. Re-install: python3 setup.py develop

The detailed description of training/test procedure can be found in [GETTING_STARTED](GETTING_STARTED.md).