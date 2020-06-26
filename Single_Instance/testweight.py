import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import DataProcess
import numpy as np
import os
import evaluate


batch_size = 32
num_classes = 10

valid_filename = 'tv.txt'
valid_data = DataProcess.ImageDataset(valid_filename)
valid_data = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle = False)


def Dic_estab(filename):
    myfile = open(filename,"r")
    table = {}
    for index,line in enumerate(myfile.readlines()):
        if line != '\n':
            table[index] = line.split('/')[2]
        #     print(index,line.split('/')[2])
    return table
    
table = Dic_estab(filename='tv.txt')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def threshold_tensor_batch(pred, base=0.50):
    p_max = torch.max(pred, dim=1)[0]
    pivot = torch.cuda.FloatTensor([base]).expand_as(p_max)
    threshold = torch.min(p_max, pivot) 
    pred_label = torch.ge(pred, threshold.unsqueeze(dim=1))
    return pred_label

def weight_cal(arr,per = 0.3):
    arr.astype(int)
    val = arr.shape[0] * per
 #   result = [0,0,0,0,0,0,0,0,0,0]
    result = np.sum(arr,axis = 0)
    for i in range(10):
        #print(result[i])
        result[i] = int(result[i] > val)
    return result
# print(weight_cal(array,0.4))


for m in [11,12,13,14,15,16]:
        valid_loss = 0.0
        valid_auc = 0.0
        valid_macro_f1 = 0.0
        valid_micro_f1 = 0.0


        count_valid = 0
        loss_function = nn.BCELoss()


        train_pred = []
        train_true = []

        epoch = 0
        count = 0    
        model = torch.load("not_freeze_models/resnet{}.pt".format(m)).to(device)           
        #model = torch.load("final_models/resnet{}.pt".format(m)).to(device)
        for valid_inputs,labels in valid_data:
                # if (labels.size(0) != 32):
                #         continue
            
                # count_valid += 1
                # if count_valid == 5:
                #     break

                valid_inputs = valid_inputs.transpose(1,3).transpose(2,3).to(device).float()
                labels = labels.to(device).float()
                outputs = model(valid_inputs)
                loss = loss_function(outputs, labels)
                valid_loss += loss.item() 
                # res = torch.gt(outputs,0.5).int()
                res = threshold_tensor_batch(outputs)
                train_pred.append(res.cpu().numpy().tolist())
                train_true.append(labels.cpu().numpy().tolist())
               # print(np.array(train_true).shape)

                # print(train_true.shape)
        train_pred = np.array(train_pred)
        train_true = np.array(train_true)
        # print(train_pred.shape)
        # print(train_true.shape)
        train_pred = train_pred.reshape(-1,10)
        train_true = train_true.reshape(-1,10)
        di = {}
        di_true = {}
        for i in range(train_pred.shape[0]):
            di_true[table[i]] = train_true[i]
            if table[i] in di.keys():
                di[table[i]]= np.concatenate( (di[table[i]], [train_pred[i]]),axis = 0)    
            else:
                di[table[i]] =[train_pred[i]]
       


        train_true = []
        train_pred = []
        for key in di.keys():
                # print(di_true[key])
                # print(weight_cal(np.array(di[key]),0.4))
     
                train_true.append(di_true[key].tolist())
                train_pred.append(weight_cal(np.array(di[key]),0.4))
        # input()
        train_pred = np.array(train_pred)
        train_true = np.array(train_true)
        print(train_pred.shape)
        print(train_true.shape)

        
        valid_auc +=  evaluate.auc(train_true ,train_pred)
        valid_macro_f1 += evaluate.macro_f1(train_true ,train_pred)
        valid_micro_f1 += evaluate.micro_f1(train_true ,train_pred)
        

                # try:
                #     valid_auc +=  evaluate.auc(labels.cpu(),res.cpu())
                #     valid_macro_f1 += evaluate.macro_f1(labels[0].cpu(),res[0].cpu())
                #     valid_micro_f1 += evaluate.micro_f1(labels[0].cpu(),res[0].cpu())
                #     count_valid += 1
                # except:
                #     print("error ",count_valid)
                #     # if (count_valid> 32):
                #     #     break
        print("valid ---- epoch == > {},loss == {}, auc ==> {},  macro_f1==> {}, micro_f1 ==> {}".format(epoch, valid_loss, valid_auc ,valid_macro_f1, valid_micro_f1))