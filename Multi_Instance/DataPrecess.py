import csv
import os
import random
import pickle as pkl

class DataSetSplit():
    def __init__(self, DataSetDir="Dataset", LabelFile="train.csv", ExtractLabelFile="Label.pkl", TrainFile="Train.pkl", TestFile="Test.pkl",
                TrainRatio = .8):
        self.TrainRatio = TrainRatio
        self.DataSetDir = DataSetDir
        self.LabelFile = os.path.join(DataSetDir, LabelFile)
        self.CurrentDir = os.path.dirname(os.path.abspath(__file__))
        self.TrainFile = os.path.join(self.CurrentDir, TrainFile)
        self.TestFile = os.path.join(self.CurrentDir, TestFile)
        self.ExtractLabelFile = os.path.join(self.CurrentDir, ExtractLabelFile)

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
        Test = []
        for bag in os.listdir(os.path.join(self.DataSetDir, "train")):
            images = [os.path.join(self.DataSetDir, "train", bag, image) for image in os.listdir(os.path.join(self.DataSetDir, "train", bag))]
            random.shuffle(images)
            Total = len(images)
            Train.append([bag, images[:int(Total*self.TrainRatio)]])
            Test.append([bag, images[int(Total*self.TrainRatio):]])
        with open(self.TrainFile, "wb") as f:
            pkl.dump(Train, f)
        with open(self.TestFile, "wb") as f:
            pkl.dump(Test, f)
        


if __name__=="__main__":
    dss = DataSetSplit()
    dss.LoadAndSplit()

