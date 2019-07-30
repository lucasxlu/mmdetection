import os
import sys
import random
import shutil

import cv2

from lxml import etree


def split_train_val_test_detection_data(xml_dir):
    """
    prepare train/val/test dataset for detection
    :param xml_dir:
    :return:
    """
    filenames = [_.replace('.xml', '') for _ in os.listdir(xml_dir)]
    random.shuffle(filenames)

    TEST_RATIO = 0.2

    train = filenames[0:int(len(filenames) * (1 - TEST_RATIO))]
    test = filenames[int(len(filenames) * (1 - TEST_RATIO)) + 1:]

    val = train[0:int(len(train) * 0.1)]
    train = train[int(len(train) * 0.1) + 1:]

    with open('./train.txt', mode='wt', encoding='utf-8') as f:
        f.writelines('\n'.join(train))

    with open('./val.txt', mode='wt', encoding='utf-8') as f:
        f.writelines('\n'.join(val))

    with open('./test.txt', mode='wt', encoding='utf-8') as f:
        f.writelines('\n'.join(test))


def mkdir_if_not_exist(dir_name):
    """
    make directory if not exist
    :param dir_name:
    :return:
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def crop_subregion_from_img(label_me_xml_path, img_dir, save_to_base_dir):
    """
    crop annotated bbox from image
    """
    mkdir_if_not_exist(save_to_base_dir)
    if label_me_xml_path.endswith('.xml'):
        print('processing %s...' % label_me_xml_path)
        tree = etree.parse(label_me_xml_path)
        objects = tree.xpath('//object')
        label_me_xml_name = label_me_xml_path.split(os.path.sep)[-1]

        img_path = os.path.join(img_dir, label_me_xml_name[0: -4] + '.jpg')
        img = cv2.imread(img_path)

        for i, object in enumerate(objects):
            sub_region = img[int(object.xpath('bndbox/ymin/text()')[0]): int(object.xpath('bndbox/ymax/text()')[0]),
                         int(object.xpath('bndbox/xmin/text()')[0]): int(object.xpath('bndbox/xmax/text()')[0])]

            print('current bbox is %s.' % object.xpath('name/text()')[0])
            bbox_obj_class = object.xpath('name/text()')[0]

            mkdir_if_not_exist(os.path.join(save_to_base_dir, bbox_obj_class))
            cv2.imwrite(os.path.join(save_to_base_dir, bbox_obj_class,
                                     "{0}_{1}_{2}.jpg".format(label_me_xml_name[0: -4], bbox_obj_class, i)), sub_region)
    
    print('done!')
    

def remove_redundant_xmls_or_jpgs(xml_dir, jpg_dir):
    """
    remove redundant xmls or images
    """
    xml_ids = [_ for _ in os.listdir(xml_dir) if _.endswith('.xml')]
    jpg_ids = [_ for _ in os.listdir(jpg_dir) if _.endswith('.jpg')]

    if len(xml_ids) < len(jpg_ids):
        for jpg_id in jpg_ids:
            if not os.path.exists(os.path.join(xml_dir, jpg_id.replace('.jpg', '.xml'))):
                os.remove(os.path.join(jpg_dir, jpg_id))
                print('deleting {}'.format(os.path.join(jpg_dir, jpg_id)))
    elif len(xml_ids) > len(jpg_ids):
        for xml_id in xml_ids:
            if not os.path.exists(os.path.join(jpg_dir, xml_id.replace('.xml', '.jpg'))):
                os.remove(os.path.join(xml_dir, xml_id))
                print('deleting {}'.format(os.path.join(xml_dir, xml_id)))

    print('done!')
