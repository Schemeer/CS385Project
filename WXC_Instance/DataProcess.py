import csv
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
#import image_process
# from utils import image_processing
from PIL import Image
import os
import random


class DataProceess():
    def __init__(self,trainFold = 8,testFold = 2):
        self.trainFold = trainFold
        self.testFold = testFold


    def produce_txt(self):
        counts = [0 for _ in range(10)]
        with open("../DataSet/train.csv",'r') as f:
            myfile = open("./Total.txt","w")
            reader = csv.reader(f)
            print(type(reader))
            for row in reader:
                path = os.path.join("../DataSet/train",row[0])

                # if len(row[1]) == 1:
                #     row[1] = row[1]+";-1;-1"
                # if len(row[1]) == 3:
                #     row[1] = row[1]+";-1"
                labels = row[1].split(';')
                # if len(labels) > 1:
                #     continue

                l= [0 for i in range(10)]
                for i in labels:
                    l[int(i)] = 1
                    # counts[int(i)] += 1 * len(os.listdir(path))

                str1 = ''
                for i in l:
                    str1 += ' '
                    str1 += str(i)

                for (root, dirs, files) in os.walk(path):
                    imagepath = [os.path.join(root,i) for i in files]
                    fileinfo = [i + str1 for i in imagepath]
                    #print(fileinfo[0])
                    for i in fileinfo:
                        myfile.write(i)
                        myfile.write('\n')
        print(counts)


    def Fold(self):
        file1 = open("./Total.txt")
        files = file1.readlines()
        length = len(files) - 1
        # print(files)
        rate = int(length * float(self.trainFold)/(self.trainFold+self.testFold))
        print(rate)
        print(length)
        shuffle = [i for i in range(length)]
        # print(shuffle)
        random.shuffle(shuffle)
        train = shuffle[:rate]
        test = shuffle[:rate]

        nf = open("./tr.txt",'w')
        nt = open("tem.txt", 'w')
  
        for i in train:
            nf.write(files[i])
    
        for i in test:
            nt.write(files[i])
  

test = DataProceess()

test.produce_txt()
test.Fold()


#produce_txt()




class ImageDataset(Dataset):
    def __init__(self, filename, resize_height = None, resize_width = None ,repeat = 1):
        self.image_label_list = self.read_file(filename)
        #self.image_dir = image_dir
        self.len = len(self.image_label_list) - 1
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width

    def __len__(self):
        if self.repeat == None:
            data_len = 1000000
        else:
            data_len = len(self.image_label_list) * self.repeat
        return data_len

    def read_file(self,filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                content = line.rstrip().split(' ')
                name = content[0]
                labels = []
                for value in content[1:]:
                    labels.append(int(value))
                image_label_list.append((name, labels))
        return image_label_list

    def load_data(self, path, resize_height, resize_width, normalization):
        #image = image_processing.read_image(path, resize_height, resize_width, normalization)
        image = Image.open(path)
        image = np.array(image)
       # print(image)
        return image


    def __getitem__(self,i):
        index = i%self.len
        image_path, label = self.image_label_list[index]
        img = self.load_data(image_path, self.resize_height, self.resize_width, normalization=False)
        #img = self.data_preproccess(img)
        label=np.array(label)
        return img,label


        


if __name__ == "__main__":
    train_filename = "./Total.txt"

    # epoch_num = 2
    # batch_size = 3000
    # # train_data_nums = 1000
    # # max_interate = int((train_data_nums+batch_size-1)/batch_size*epoch_num)
    # train_data = ImageDataset(train_filename)
    # train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle = False)
    # for epoch in range(epoch_num):
    #     for batch_image, batch_label in train_loader:
    #         print(batch_image.shape)
    #         print(batch_label.shape)
#             print("batch_image.shape:{},batch_label:{}".format(batch_image.shape,batch_label))
