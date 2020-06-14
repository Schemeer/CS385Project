import csv
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import pickle as pkl
import random

class ImageDataset(Dataset):
    def __init__(self, LabelFile="Label.pkl", DataFile="Train.pkl", resize_height=512, resize_width=512, repeat=1, classnum=10, transform=None):
        self.CurrentDir = os.path.dirname(os.path.abspath(__file__))
        self.Labels = self.read_file(os.path.join(self.CurrentDir, LabelFile))
        self.Datas = self.read_file(os.path.join(self.CurrentDir, DataFile))
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.classnum = classnum
        self.transform = transform

    def __len__(self):
        if self.repeat == None:
            data_len = 1000000
        else:
            data_len = len(self.Datas) * self.repeat
        return data_len

    def read_file(self, filename):
        with open(filename, "rb") as f:
            res = pkl.load(f)
        return res

    def load_data(self, paths, resize_height, resize_width, normalization):
        images = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            image = self.transform(image).numpy()
            images.append(image)
        return torch.from_numpy(np.array(images))

    def encode(self, bag):
        label_ls = self.Labels[bag]
        label = np.zeros(self.classnum)
        label[label_ls]=1
        return torch.from_numpy(label).float()

    def __getitem__(self,i):
        index = i%len(self.Datas)
        bag, image_paths = self.Datas[index]
        imgs = self.load_data(image_paths, self.resize_height, self.resize_width, normalization=False)
        label = self.encode(bag)
        return imgs, label

class KfoldDataset(Dataset):
    def __init__(self, LabelFile="Label.pkl", FoldFile="Fold", FoldIds=[0,1,2,3], resize_height=512, resize_width=512, repeat=1, classnum=10, transform=None):
        self.CurrentDir = os.path.dirname(os.path.abspath(__file__))
        self.Labels = self.read_file(os.path.join(self.CurrentDir, LabelFile))
        self.Datas = self.read_file(os.path.join(self.CurrentDir, FoldFile), FoldIds)
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.classnum = classnum
        self.transform = transform
        

    def __len__(self):
        if self.repeat == None:
            data_len = 1000000
        else:
            data_len = len(self.Datas) * self.repeat
        return data_len

    def read_file(self, filename, ids=None):
        res = []
        if ids:
            for i in ids:
                with open(filename+f"_{i}.pkl", "rb") as f:
                    res.extend(pkl.load(f))
        else:
            with open(filename, "rb") as f:
                res = pkl.load(f)
        return res

    def load_data(self, paths, resize_height, resize_width, normalization):
        images = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            image = self.transform(image).numpy()
            images.append(image)
        return torch.from_numpy(np.array(images))

    def encode(self, bag):
        label_ls = self.Labels[bag]
        label = np.zeros(self.classnum)
        label[label_ls]=1
        return torch.from_numpy(label).float()

    def __getitem__(self,i):
        index = i%len(self.Datas)
        bag, image_paths = self.Datas[index]
        imgs = self.load_data(image_paths, self.resize_height, self.resize_width, normalization=False)
        label = self.encode(bag)
        return imgs, label


if __name__ == "__main__":
    # epoch_num = 2
    # batch_size = 1
    # train_data = ImageDataset()
    # train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle = False)
    # for epoch in range(epoch_num):
    #     for batch_image, batch_label in train_loader:
    #         print(batch_image.shape)
    #         print(batch_label.shape)
    
    train_data = KfoldDataset(FoldIds=[0,1,2,3])
    test_data = KfoldDataset(FoldIds=[4])
    print(len(train_data), len(test_data))