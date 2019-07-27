import copy
import json
import os
import random
import sys
import shutil

import cv2
from lxml import etree


def merge_multi_labels_to_one(voc_xml_dir, save_to_dir, one_label_name):
    """
    merge all cls labels to one
    :param voc_xml_dir:
    :param one_label_name:
    :return:
    """
    if not os.path.exists(save_to_dir):
        os.makedirs(save_to_dir)

    for xml in os.listdir(voc_xml_dir):
        if xml.endswith('.xml'):
            tree = etree.parse(os.path.join(voc_xml_dir, xml))
            objects = tree.xpath('//object')
            for object in objects:
                object.xpath('name')[0].text = one_label_name

            tree.write(os.path.join(save_to_dir, xml), pretty_print=True, xml_declaration=True, encoding="utf-8")
            print('convert {0} successfully'.format(xml))

    print('done!')


def cvt_yolo2voc_format(labelme_yolo_path, obj_dict, img_dir, convert_to_dir='C:/Users/Administrator/Desktop/'):
    """
    convert YOLO format to Pascal VOC format
    :param labelme_yolo_path:
    :param obj_dict:
    :param img_dir:
    :param convert_to_dir:
    :return:
    """
    if labelme_yolo_path.endswith('.txt'):
        print("processing %s" % labelme_yolo_path)
        img_h, img_w, img_c = cv2.imread(
            os.path.join(img_dir, labelme_yolo_path.split('/')[-1].replace('.txt', '.jpg'))).shape

        annotation = etree.Element("annotation")
        etree.SubElement(annotation, "filename").text = labelme_yolo_path.split('/')[-1].replace('.txt', '.jpg')
        size = etree.SubElement(annotation, "size")
        etree.SubElement(size, "width").text = str(img_w)
        etree.SubElement(size, "height").text = str(img_h)
        etree.SubElement(size, "depth").text = str(img_c)

        with open(labelme_yolo_path, mode='rt', encoding='utf-8') as f:
            for _ in f.readlines():
                if _.strip() is not "":
                    cls_name = obj_dict[int(_.split(" ")[0].strip())]
                    bbox_w = float(_.split(" ")[3].strip()) * img_w
                    bbox_h = float(_.split(" ")[4].strip()) * img_h
                    xmin = int(float(_.split(' ')[1]) * img_w - bbox_w / 2)
                    xmax = int(float(_.split(' ')[1]) * img_w + bbox_w / 2)
                    ymin = int(float(_.split(' ')[2]) * img_h - bbox_h / 2)
                    ymax = int(float(_.split(' ')[2]) * img_h + bbox_h / 2)

                    object = etree.SubElement(annotation, "object")
                    etree.SubElement(object, "name").text = cls_name
                    etree.SubElement(object, "difficult").text = 0
                    bndbox = etree.SubElement(object, "bndbox")
                    etree.SubElement(bndbox, "xmin").text = str(xmin)
                    etree.SubElement(bndbox, "ymin").text = str(ymin)
                    etree.SubElement(bndbox, "xmax").text = str(xmax)
                    etree.SubElement(bndbox, "ymax").text = str(ymax)

            tree = etree.ElementTree(annotation)
            tree.write(os.path.join(convert_to_dir, '{0}'.format(
                labelme_yolo_path.split('/')[-1].replace('.txt', '.xml'))), pretty_print=True,
                       xml_declaration=True, encoding="utf-8")
            print('write %s successfully.' % labelme_yolo_path.split('/')[-1].replace('.txt', '.xml'))
    
    print('done!')