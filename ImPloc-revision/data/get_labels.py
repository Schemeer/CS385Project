import csv
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import random


def produce_txt():
    with open("train.csv", 'r') as f:
        myfile = open("enhanced1.list", "w")
        reader = csv.reader(f)
        for row in reader:
            labels = row[1].split(';')
            # l = [0 for i in range(10)]
            # for i in labels:
            #     l[int(i)] = 1
            # str1 = ''
            # for i in l:
            #     str1 += ' '
            #     str1 += str(i)
            str1 = ','.join(labels)
            # print(row[0] + ' ' + str1)
            myfile.write(row[0])
            myfile.write('\n')

            # path = os.path.join("train", row[0])
            # for (root, dirs, files) in os.walk(path):
            #     imagepath = files
            #     print(imagepath)
                # fileinfo = [i + str1 for i in imagepath]
                # print(fileinfo[0])
                # for i in fileinfo:
                #     myfile.write(i)
                #     myfile.write('\n')


if __name__ == '__main__':
    produce_txt()