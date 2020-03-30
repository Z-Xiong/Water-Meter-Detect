import os
import os.path

import numpy as np

import cv2
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import torch

path_to_img = "C:/Users/10138/Documents/yottacloud/code/water-meter-detect/data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/"

def read_images(root, train=True):
    txt_fname = root+('VOC2007')+('train.txt' if train else 'test.txt')
    with open(txt_fname) as f:
        lines = f.readlines()

    fnames = []
    boxes = []
    labels = []
    for line in lines:
        splited = line.split()
        fnames.append(splited[0])
        num_boxes = (len(splited) - 1) // 5
        box = []
        label = []
        for i in range(num_boxes):
            x = float(splited[1 + 5 * i])
            y = float(splited[2 + 5 * i])
            x2 = float(splited[3 + 5 * i])
            y2 = float(splited[4 + 5 * i])
            c = splited[5 + 5 * i]
            box.append([x, y, x2, y2])
            label.append(int(c) + 1)
        boxes.append(torch.Tensor(box))
        labels.append(torch.LongTensor(label))

    return fnames, boxes, labels


def encoder(boxes, labels):
    """
    boxes (tensor) [[x1,y1,x2,y2],[]]
    labels (tensor) [...]
    return 7x7x30
    """
    grid_num = 7
    target = torch.zeros((grid_num, grid_num, 30))
    cell_size = 1. / grid_num
    wh = boxes[:, 2:] - boxes[:, :2]
    cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2
    for i in range(cxcy.size()[0]):
        cxcy_sample = cxcy[i]
        ij = (cxcy_sample / cell_size).ceil() - 1  #
        target[int(ij[1]), int(ij[0]), 4] = 1
        target[int(ij[1]), int(ij[0]), 9] = 1
        target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1
        xy = ij * cell_size  # 匹配到的网格的左上角相对坐标
        delta_xy = (cxcy_sample - xy) / cell_size
        target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
        target[int(ij[1]), int(ij[0]), :2] = delta_xy
        target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
        target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
    return target

def subMean(bgr, mean):
    mean = np.array(mean, dtype=np.float32)
    bgr = bgr - mean
    return bgr


class WaterMeterDataset(Dataset):
    """
    txt描述文件 image_name.jpg x y w h c x y w h c 这样就是说一张图片中有两个目标
    """
    def __init__(self, root, trsfm, train=True):
        self.root = root
        self.train = train
        self.tranform = trsfm
        self.fnames, self.boxes, self.labels = read_images(root, train=self.train)
        self.num_samples = len(self.boxes)
        self.mean = (123, 117, 104)  # RGB

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(path_to_img+fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        # # debug
        # box_show = boxes.numpy().reshape(-1)
        # print(box_show)
        # img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # pt1 = (int(box_show[0]), int(box_show[1]))
        # pt2 = (int(box_show[2]), int(box_show[3]))
        # cv2.rectangle(img_show, pt1=pt1, pt2=pt2, color=(0, 255, 0), thickness=1)
        # plt.figure()
        #
        # cv2.rectangle(img, pt1=(10, 10), pt2=(100, 100), color=(0, 255, 0), thickness=1)
        # plt.imshow(img_show)
        # plt.show()
        # # debug

        h, w, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = subMean(img, self.mean)
        img = Image.fromarray(np.uint8(img))
        img = self.tranform(img)
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        target = encoder(boxes, labels)

        return img, target

    def __len__(self):
        return self.num_samples


class WaterMeterDataLoader(BaseDataLoader):
    """
    Water-Meter data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = WaterMeterDataset(self.data_dir, trsfm, train = training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
