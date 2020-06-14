import csv
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import random

DATA_DIR = '../DataSet/'

class DataProceess():
    def __init__(self,trainFold = 7,validFold = 1, testFold = 2):
        self.validFold = validFold
        self.trainFold = trainFold
        self.testFold = testFold


    def produce_txt(self):
        with open(DATA_DIR + "train.csv",'r') as f:
            myfile = open(DATA_DIR + "Total.txt","w")
            reader = csv.reader(f)
            for row in reader:
                row[1] = row[1].split(';')
                l= [0 for i in range(10)]
                for  i in row[1]:
                   l[int(i)] = 1
                str1 =''
                for i in l:
                    str1 += ' '
                    str1 += str(i)
     
    
                path = os.path.join(DATA_DIR + "train",row[0])
                for (root, dirs, files) in os.walk(path):
                    imagepath = [os.path.join(root, i) for i in files]
                    print(imagepath)
                    fileinfo = [i + str1 for i in imagepath]
                    # print(fileinfo[0])
                    for i in fileinfo:
                        myfile.write(i)
                        myfile.write('\n')

    def Fold(self):
        file1 = open(DATA_DIR + "Total.txt")
        files = file1.readlines()
        length = len(files) - 1
 
        rate1 = int(length * float(self.trainFold)/(self.trainFold+self.testFold+self.validFold))
        rate2 = int(length * float(self.trainFold+self.validFold)/(self.trainFold+self.testFold+self.validFold))

        print("rate1:{} ,rate2:{}".format(rate1,rate2))
        shuffle = [i for i in range(length)]

        random.shuffle(shuffle)

        train = shuffle[:rate1]
        valid = shuffle[rate1:rate2]
        test = shuffle[rate2:]

        nf = open(DATA_DIR + "tr.txt", 'w')
        nv = open(DATA_DIR + "tv.txt", 'w')
        nt = open(DATA_DIR + "te.txt", 'w')

        for i in train:
            nf.write(files[i])
  
        for i in valid:
            nv.write(files[i])
    
        for i in test:
            nt.write(files[i])
  


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
            data_len = 10000
        else:
            data_len = self.len * self.repeat
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
        image = Image.open(path)#.conver("L")
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
    test = DataProceess()
    test.produce_txt()
    test.Fold()

    # train_filename = "Total.txt"
    #
    # epoch_num = 1
    # batch_size = 3000
    # # train_data_nums = 1000
    # # max_interate = int((train_data_nums+batch_size-1)/batch_size*epoch_num)
    # count = 0
    # train_data = ImageDataset(train_filename)
    # train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle = False)
    # for epoch in range(epoch_num):
    #     for batch_image, batch_label in train_loader:
    #         count+=1
    #         print(batch_image.shape)
    #         print(batch_label.shape)
    #         print(count)
#             print("batch_image.shape:{},batch_label:{}".format(batch_image.shape,batch_label))
