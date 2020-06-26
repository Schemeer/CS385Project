import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from Multi_Instance import DataSet
from Multi_Instance import MIMLModel
from Multi_Instance import DataProcess
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import csv


DATASETDIR = "unknown" # 自己填一下
FOLD_FILE = "TEST"
GENERATE_FOLDFILE = True
GROUP_SIZE = 8
BATCH_SIZE = 4
MODEL_DIR = "unknown" # 自己填一下
MODEL_FILE = "unknoen" # 自己填一下
TEST_FOLD = [0]
LABEL_FILE = "Label.csv"
PROB_FILE = "Prob.csv"
DELIMITER = ";"
RESULT_DIR = "Result"

def threshold_tensor_batch(pred, base=0.5):
    p_max = torch.max(pred, dim=1)[0]
    pivot = torch.cuda.FloatTensor([base]).expand_as(p_max)
    threshold = torch.min(0.9*p_max, pivot)
    pred_label = torch.ge(pred, threshold.unsqueeze(dim=1))
    return pred_label


if __name__=="__main__":
    if GENERATE_FOLDFILE:
        Dprocess = DataProcess.KfoldDataSet(DataSetDir=DATASETDIR, NumFold=1, FoldFile=FOLD_FILE)
        Dprocess.LoadAndSplit()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MIMLModel.ImPlocMIML(group_size=GROUP_SIZE)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, MODEL_FILE)))
    model.to(device)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = DataSet.KfoldDataset(transform=transform, FoldIds=TEST_FOLD)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
    prob = [] # 概率
    pred = [] # 标签
    bags = [] # 包名
    model.eval()
    for batch_image, batch_label, batch_bag in dataloader:
        batch_image = batch_image.view(-1,3,512,512).to(device)
        batch_label = batch_label

        output = model(batch_image)

        res = threshold_tensor_batch(output, 0.5)

        prob.append(output.cpu().numpy())
        pred.append(res.cpu().numpy())
        bags.extend(batch_bag)

    prob = np.concatenate(prob)
    pred = np.concatenate(pred)

    '''
    整合成包的结果
    '''

    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    label_file = open(os.path.join(RESULT_DIR, LABEL_FILE), "w", newline="")
    label_writer = csv.writer(label_file)
    prob_file = open(os.path.join(RESULT_DIR, PROB_FILE), "w", newline="")
    prob_writer = csv.writer(prob_file)
    for i in range(len(bags)):
        label = DELIMITER.join([str(predij[0]) for predij in np.argwhere(pred[i]==True)])
        label_writer.writerow([bags[i],label])
        prob = DELIMITER.join([f"{probij:.4f}" for probij in prob[i]])
        prob_writer.writerow([bags[i],prob])
    label_file.close()
    prob_file.close()

