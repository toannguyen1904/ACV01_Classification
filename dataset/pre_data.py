import numpy as np
import imageio
import os
from PIL import Image
from torchvision import transforms
import random


class Dataset():
    def __init__(self, input_size, data_root, train_folder_path, val_folder_path, mode='train'):
        self.input_size = input_size
        self.data_root = data_root
        self.mode = mode

        train_info = open(os.path.join(self.data_root, 'meta', 'train.txt'))
        val_info = open(os.path.join(self.data_root, 'meta', 'val.txt'))

        train_img_label = []
        val_img_label = []
        for line in train_info:
            train_img_label.append([os.path.join(train_folder_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])])
        for line in val_info:
            val_img_label.append([os.path.join(val_folder_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])])

        random.shuffle(train_img_label)
        random.shuffle(val_img_label)

        self.train_img_label = train_img_label[:60000]
        self.val_img_label = val_img_label[:15000]

    def __getitem__(self, index):
        if self.mode == 'train':
            img, target = imageio.imread(self.train_img_label[index][0]), self.train_img_label[index][1]
    
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize(self.input_size + 16)(img)  # old 16
            img = transforms.RandomRotation(20)(img)
            img = transforms.RandomVerticalFlip()(img)
            img = transforms.RandomCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        elif self.mode == 'val':
            img, target = imageio.imread(self.val_img_label[index][0]), self.val_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize(self.input_size + 16)(img)  # old 16
            img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_img_label)
        elif self.mode == 'val':
            return len(self.val_img_label)