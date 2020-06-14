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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for m in [10,11,12,13,14,15,16,17,18,19]:
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
        #model = torch.load("testmodels/resnet{}.pt".format(m)).to(device)           
        model = torch.load("not_freeze_models/resnet{}.pt".format(m)).to(device)
        for valid_inputs,labels in valid_data:
                if (labels.size(0) != 32):
                        continue
                # count_valid +=1
                # if count_valid == 5:
                #     break
                valid_inputs = valid_inputs.transpose(1,3).transpose(2,3).to(device).float()
                labels = labels.to(device).float()
                outputs = model(valid_inputs)
                loss = loss_function(outputs, labels)
                valid_loss += loss.item() 
                res = torch.gt(outputs,0.5).int()
                train_pred.append(res.cpu().numpy().tolist())
                train_true.append(labels.cpu().numpy().tolist())
                #print(np.array(train_true).shape)

                # print(train_true.shape)
        train_pred = np.array(train_pred)
        train_true = np.array(train_true)
        # print(train_pred.shape)
        # print(train_true.shape)
        train_pred = train_pred.reshape(-1,10)
        train_true = train_true.reshape(-1,10)
        # print(train_pred.shape)
        
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