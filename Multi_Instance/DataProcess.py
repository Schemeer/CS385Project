import csv
import os
import random
import pickle as pkl

class DataSetSplit():
    def __init__(self, DataSetDir="Dataset", LabelFile="train.csv", ExtractLabelFile="Label.pkl", TrainFile="Train.pkl", ValidFile="Valid.pkl",
                TrainRatio = .8, GroupSize=8):
        self.TrainRatio = TrainRatio
        self.DataSetDir = DataSetDir
        self.LabelFile = os.path.join(DataSetDir, LabelFile)
        self.CurrentDir = os.path.dirname(os.path.abspath(__file__))
        self.TrainFile = os.path.join(self.CurrentDir, TrainFile)
        self.ValidFile = os.path.join(self.CurrentDir, ValidFile)
        self.ExtractLabelFile = os.path.join(self.CurrentDir, ExtractLabelFile)
        self.GroupSize = GroupSize

    def LoadAndSplit(self):
        # Load bag&label information
        bag2label = {}
        str2int = {str(i):i for i in range(10)}
        with open(self.LabelFile,'r') as f:
            reader = csv.reader(f)
            for row in reader:
                labels = [str2int[i] for i in row[1].split(";")]
                bag2label[row[0]] = labels
        with open(self.ExtractLabelFile, "wb") as f:
            pkl.dump(bag2label, f)
        
        Train = []
        Valid = []
        count = 0
        for bag in os.listdir(os.path.join(self.DataSetDir, "train")):
            images = [os.path.join(self.DataSetDir, "train", bag, image) for image in os.listdir(os.path.join(self.DataSetDir, "train", bag))]
            random.shuffle(images)
            Total = len(images)
            resnum = 8 - Total % 8
            count += Total
            images.extend(images[:resnum])
            Traincount = int(len(images)//8*self.TrainRatio)
            
            images = [[bag, images[i*8:i*8+8]] for i in range(len(images)//8)]
            Train.extend(images[:Traincount])
            Valid.extend(images[Traincount:])
        random.shuffle(Train)
        random.shuffle(Valid)
        with open(self.TrainFile, "wb") as f:
            pkl.dump(Train, f)
        with open(self.ValidFile, "wb") as f:
            pkl.dump(Valid, f)
        


if __name__=="__main__":
    dss = DataSetSplit()
    dss.LoadAndSplit()

