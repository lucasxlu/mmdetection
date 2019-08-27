import os

import mmdet
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result, init_detector


def eval_mmdet(config_py='configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py', pretrained_model='work_dirs/faster_rcnn_r50_fpn_1x_voc0712/epoch_100.pth', 
    img_dir='data/VOCdevkit/VOC2007/JPEGImages'):
    config_file = config_py
    checkpoint_file = pretrained_model
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # test a list of images
    with open('data/VOCdevkit/VOC2007/ImageSets/Main/test.txt', mode='rt', encoding='utf-8') as f:
        img_ids = f.readlines()
    imgs = [os.path.join(img_dir, "{}.jpg".format(img_id.replace('\n', ''))) for img_id in img_ids]
    print(imgs)
    for i, result in enumerate(inference_detector(model, imgs)):
        if not os.path.exists('./eval_mmdet_imgs'):
            os.makedirs('./eval_mmdet_imgs')
        
        print(i, imgs[i])    
        show_result(imgs[i], result, model.CLASSES, out_file=os.path.join('./eval_mmdet_imgs', imgs[i].split('/')[-1]))

if __name__ == '__main__':
    # eval_mmdet(config_py='configs/pascal_voc/ssd512_voc.py', pretrained_model='work_dirs/ssd512_voc/epoch_30.pth')
    eval_mmdet(config_py='configs/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_tissuephysiology.py', pretrained_model='work_dirs/libra_faster_rcnn_r50_fpn_1x_TissuePhysiology/epoch_300.pth')
