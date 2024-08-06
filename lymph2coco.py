import numpy as np
import os 
import torchio as tio
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import json
from pathlib import Path
from PIL import Image, ImageDraw
import random
random.seed(42)

# This file for turn the 3D formet of myself dataset to 2D coco format !!!!!!!!!!!!!

# Let's store the final annotations in CSV format, with one image having multiple labels.
# Use the image ID in the format name-0-slice number.

# After reading the image, slice it, generate the corresponding labels, and then read them.
# For label conversion, although there might be overlap, let's ignore it! 
# Directly read and convert nnUNet images and labels for labels or other data.

# 最后annotation也用csv形式储存吧，可以一张图片然后很多的label这种
# 这个image id就使用这个 name-0-slice数

# 读取image之后再切成slice，然后再把相对应的label给造出来，然后再来读取这种，
# 对于label的读取转化，虽然可能会有overlap的情况，但是我们忽略吧！！！对于label或者什么直接读取nnunet的image和label来进行转化吧

def arr2rgb(array, name, slice, label_boxes, data_type, save=False):
    rgb_image = Image.new("RGB", (array.shape[1], array.shape[0]))
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            pixel_value = array[y, x]
            rgb_image.putpixel((x, y), (pixel_value, pixel_value, pixel_value))  # 设置为灰度色彩

    if save and np.max(label_boxes) != 0:
        draw = ImageDraw.Draw(rgb_image)

        for rect in label_boxes:
            left, top, width, height = rect
            # print(f'the width, height is {width, height}')
            draw.rectangle([(left, top), (left + width, top + height)], outline=(255, 0, 0))  # 在图像上绘制矩形框，红色

    rgb_image.save(f"/public_bme/data/xiongjl/Deformable-DETR_data_public/coco/{data_type}2017_alllayers/{name}_{slice}.jpg")


def label_mask2x1y1wh(array):
    array = np.array(array, dtype=np.int8)
    drop = 1
    save = True
    positive = True
    if array.max() == 1: 
        # 标记连通域
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(array, connectivity=4)
        # 提取每个连通域的外接长方形坐标
        rectangles = []
        for label in range(1, num_labels):
            left = stats[label, cv2.CC_STAT_LEFT]
            top = stats[label, cv2.CC_STAT_TOP]
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            if (width + height) / 2 > 7:
                rectangles.append([left, top, width, height])
            else:
                drop += 1
        # 如果这个slice上的所有的bbox都被drop掉了，这张slice就不要了
        # print(f'drop is {drop}, num_labels is {num_labels}')
        if num_labels == drop:
            # print(f'the slice is dismiss@@@')
            save = False

        if len(rectangles) == 0:
            rectangles = [[0, 0, 0, 0]]
            positive = False
    else:
        rectangles = [[0, 0, 0, 0]]
        positive = False

    # 把这个boxes写入这个csv文件，要写的内容要包括这个 ['image_id', 'segmentation', 'box', 'area', 'category_id', 'iscrowd', id]
    return rectangles, save, positive


def bbox_to_segmentation(bbox):
    left, top, w, h = bbox
    x1, y1 = left, top
    x2, y2 = left + w, top
    x3, y3 = left + w, top + h
    x4, y4 = left, top + h
    segmentation_points = [x1, y1, x2, y2, x3, y3, x4, y4]
    return segmentation_points #.reshape((-1, 1, 2))


def image_label_slice_generate(names_list, part):
    categories = []
    categories.append({"id": 1, "name": 'lymph_node'})
    ann_dict = {}
    ann_id = 0
    ann_dict['categories'] = categories

    images = [] # 为了存整个的 whole nii image 的信息
    annotations = []
    for i, name in tqdm(enumerate(names_list)):
        if part == 'training':
            number = str(i)
            data_type = 'train'
        if part == 'validation':
            number = str(i+870)
            data_type = 'val'
        if part == 'testing':
            number = str(i+970)
        if len(number) == 1:
            number = f'00{number}'
        elif len(number) == 2:
            number = f'0{number}'
        elif len(number) == 3:
            pass
        print(f'name : {name}, the number is {number}')
        # 取一整个 3d 的 nii image
        img_number_path = f'/public_bme/data/xiongjl/nnunet2_data/nnUNet_raw/Dataset501_lymph/imagesTr/lymph_{number}_0000.nii.gz'
        label_number_path = f'/public_bme/data/xiongjl/nnunet2_data/nnUNet_raw/Dataset501_lymph/labelsTr/lymph_{number}.nii.gz'
        img = tio.ScalarImage(img_number_path)
        # * 窗宽窗位设置一下
        clamped = tio.Clamp(out_min=-160, out_max=240)
        img = clamped(img)
                                # * resample一下
                                # resample = tio.Resample((0.7, 0.7, 0.7))
                                # img = resample(clamped_img)
                                # * 归一化到 0-1 之间
                                # data_max = clamped_img.data.max()
                                # data_min = clamped_img.data.min()
                                # img = (clamped_img.data - data_min) / (data_max - data_min)
                                # print(f'img max is {img.data.max()}')
                                # print(f'img min is {img.data.min()}')
        label = tio.ScalarImage(label_number_path)
        # label = resample(label)
        shape = img.shape
        arr_img = img.data
        arr_label = label.data
        po = 0
        for slice_arr in range(0, shape[3]): #shape[3]):   # 每一张slice_arr
            
            image_slice_arr = arr_img[0, :, :, slice_arr]
            label_slice_arr = arr_label[0, :, :, slice_arr]
            # 把label的mask变成bbox的框 形式是（x1, y1, w, h）也就是 (left, top, width, height)
            # 然后还要把这个label boxes写入这个csv文件
            label_boxes, whether_save, positive = label_mask2x1y1wh(label_slice_arr)

            # if whether_save:
            #     # 把img的slice_arr变成rgb文件去储存
            #     p = random.random()
                # if positive or (not positive and p >= 0.7):
            image = {}   # 这个是为了一张二维image的信息
            image['width'] = image_slice_arr.shape[0]
            image['height'] = image_slice_arr.shape[1]
            image['file_name'] = f'{name}_{slice_arr}.jpg'
            image['id'] = int(f'{name}{slice_arr}')
            if positive:
                image['positive'] = 1
            else:
                image['positive'] = 0
            images.append(image)
            for bbox in label_boxes:
                bbox = [int(x) for x in bbox]
                # seg = bbox_to_segmentation(bbox)
                area = bbox[2] * bbox[3]
                arr2rgb(image_slice_arr, name, slice_arr, label_boxes, data_type)
                ann = {} # 这个是一张slice上有很多的boxes, 每一个的box都是有自己的anno的
                ann['id'] = ann_id
                ann_id += 1
                ann['image_id'] = image['id']
                # ann['segmentation'] = seg
                if positive:
                    ann['category_id'] = 1
                else:
                    ann['category_id'] = 2
                ann['iscrowd'] = 0
                ann['area'] = area
                ann['bbox'] = bbox
                annotations.append(ann)

    ann_dict['images'] = images
    ann_dict['annotations'] = annotations
    print("Num categories: %s" % len(categories))
    print("Num images: %s" % len(images))
    print("Num annotations: %s" % len(annotations))
    out_dir = '/public_bme/data/xiongjl/Deformable-DETR_data_public/coco/annotations'
    json_name = f'instances_{data_type}2017_alllayers_val.json'
    with open(os.path.join(out_dir, json_name), 'wb') as outfile:
        outfile.write(json.dumps(ann_dict).encode())



def main_func():
    parts = ['validation']#, ['validation']
    for part in parts:
        print(f'strating the {part} ...')
        names_list = []
        with open(f'/public_bme/data/xiongjl/nnDet/csv_files/{part}_names.csv') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == 'ZYCT201905290807' or row[0] == 'ZYCT201905010345' or row[0] == 'ZYCT201904220268':
                    pass
                else:
                    names_list.append(row[0])
        image_label_slice_generate(names_list, part)

if __name__ == "__main__":
    print(f'2coco')

    # 因为有随机数，所以一个就是json文件对应着唯一的image，但是我会给个seed之后可能就一样了
    main_func()
            