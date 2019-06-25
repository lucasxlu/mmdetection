"""
Detector
Author: LucasX
"""
import os
import time
from pprint import pprint

import PIL
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import transforms

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result
from mmdet.models import build_detector



class Detector:
    """
    Detector 
    """
    def __init__(self, pretrained_classifier_model):
        cfg = mmcv.Config.fromfile(
            'configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py')
        cfg.model.pretrained = 'work_dirs/faster_rcnn_r50_fpn_1x_voc0712/epoch_90.pth'
        
        # construct the model and load checkpoint
        model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
        _ = load_checkpoint(model, cfg.model.pretrained)
        self.detector = model

        densenet121 = models.densenet121(pretrained=False)
        num_ftrs = densenet121.classifier.in_features
        densenet121.classifier = nn.Linear(num_ftrs, 43)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        densenet121 = densenet121.to(device)
        
        state_dict = torch.load(pretrained_classifier_model)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        densenet121.load_state_dict(new_state_dict)
        densenet121.eval()

        self.classifier = densenet121
        self.device = device
        self.mapping = {}
        with open('./predefined_classes.txt', mode='rt', encoding='utf-8') as f:
            for i, itm in enumerate(f.readlines()):
                self.mapping[i] = itm.strip().replace('\n', '')

        self.cfg = cfg
        self.topK = 1
        self.crop = False
    
    def classify_from_mat(self, mat):
        """
        classify a cropped region
        """
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = preprocess(mat)
        img.unsqueeze_(0)
        img = img.to(self.device)

        outputs = self.classifier(img)
        outputs = F.softmax(outputs, dim=1)

        topK_prob, topK_label = torch.topk(outputs, self.topK)
        prob = topK_prob.to("cpu").detach().numpy().tolist()

        _, predicted = torch.max(outputs.data, 1)

        return self.mapping[int(predicted.to("cpu"))]


    def detect_and_classify(self, img_filepath):
        """
        test a single image
        """
        tik = time.time()
        result = {"status": 0, "data": {}}

        img = mmcv.imread(img_filepath)
        res = inference_detector(self.detector, img, self.cfg)
        if len(res) != 0:
            f_index = 0
            for cat, bbox in enumerate(res):
                print(cat, bbox)
                
                if len(bbox) != 0:
                    # crop patches with mmcv
                    bboxes = np.array(bbox[:, 0:-1]).astype(np.int)  # 0:-1 represents that the last col is confidence, and the first 4 cols are x1 y1 x2 y2
                    patches = mmcv.imcrop(img, bboxes)

                    for patch in patches:
                        patch = PIL.Image.fromarray(patch)
                        b, g, r = patch.split()
                        patch = Image.merge("RGB", (r, g, b))
                        if self.crop:
                            patch.save('{0}.jpg'.format(f_index))
                        cat_name = self.classify_from_mat(patch)
                        print(cat_name)
                        if cat_name in result['data'].keys():
                            result['data'][cat_name] += 1
                        else:
                            result['data'][cat_name] = 1

                        f_index += 1
                # else:
                #     if "Bad" not in result['data'].keys():
                #         result['data']["Bad"] = 1
                #     else:
                #         result['data']["Bad"] += 1
                tok = time.time()
        else:
            result['data'] = {"Bad": 1}

        result['elapse'] = tok - tik

        return result


if __name__ == '__main__':
    detector = Detector()
    result = detector.detect(img_filepath = "./test.jpg")
    pprint(result)