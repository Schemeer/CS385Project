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
train_filename = 'tr.txt'
train_data = DataProcess.ImageDataset(train_filename)
train_data = DataLoader(dataset = train_data, batch_size=batch_size, shuffle = False)


valid_filename = 'tv.txt'
valid_data = DataProcess.ImageDataset(valid_filename)
valid_data = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle = False)

print(len(train_data),len(valid_data)) 


# Load resnet
resnet50 = models.resnet18(pretrained = False)
for param in resnet50.parameters():
    param.requires_grad = True

fc_inputs = 2048
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 10),
    nn.Sigmoid()
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet50 = resnet50.to(device)
loss_func = nn.BCELoss()
optimizer = optim.Adam(resnet50.parameters(), lr = 0.00005)


def train_and_valid(model, loss_function,optimizer,epochs,train_data,valid_data,rep):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device :",device)
    history = []
    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1,epochs))
        model.train()
        train_loss = 0.0
        train_auc = 0.0
        train_macro_f1 = 0.0
        train_micro_f1 = 0.0

        valid_loss = 0.0
        valid_auc = 0.0
        valid_macro_f1 = 0.0
        valid_micro_f1 = 0.0

        count_item = 0
        count_valid = 0
        print("-----train mode-----")

        for inputs,labels in train_data:
          #  print("count item",count_item)
            inputs = inputs.transpose(1,3).transpose(2,3).to(device).float()
            labels = labels.to(device).float()
            # print(inputs.shape)
           
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item() * inputs.size(0)

            res = torch.gt(outputs,0.5)
            
            train_auc += np.sum(np.array([evaluate.auc(labels[i].cpu(),res[i].cpu()) for i in range(res.size(0))]))
            train_macro_f1 += np.sum(np.array([evaluate.macro_f1(labels[i].cpu(),res[i].cpu())  for i in range(res.size(0))]))
            train_micro_f1 += np.sum(np.array([evaluate.micro_f1(labels[i].cpu(),res[i].cpu())  for i in range(res.size(0))]))
            count_item += labels.size(0)
            # if (count_item > 128):
            #     break
        print("train ---- epoch == > {},loss==>{}, auc ==> {},  macro_f1==> {}, micro_f1 ==> {}".format(epoch,train_loss/count_item,train_auc/count_item,train_macro_f1/count_item,train_micro_f1/count_item))
        epoch_end  =time.time()
        print("time : ",epoch_end - epoch_start)
        print("-----valid mode-----")
        with torch.no_grad():
            model.eval()
        for valid_inputs,labels in valid_data:
            valid_inputs = valid_inputs.transpose(1,3).transpose(2,3).to(device).float()
            labels = labels.to(device).float()
            outputs = model(valid_inputs)
            loss = loss_function(outputs, labels)
            valid_loss += loss.item() * valid_inputs.size(0)

            res = torch.gt(outputs,0.5)
        
            valid_auc += np.sum(np.array([evaluate.auc(labels[i].cpu(),res[i].cpu()) for i in range(res.size(0))]))
            valid_macro_f1 += np.sum(np.array([evaluate.macro_f1(labels[0].cpu(),res[0].cpu())  for i in range(res.size(0))]))
            valid_micro_f1 += np.sum(np.array([evaluate.micro_f1(labels[0].cpu(),res[0].cpu())  for i in range(res.size(0))]))
            count_valid += labels.size(0)
            # if (count_valid> 128):
            #     break
        print("valid ---- epoch == > {},loss == {}, auc ==> {},  macro_f1==> {}, micro_f1 ==> {}".format(epoch, valid_loss/count_valid, valid_auc/count_valid, valid_macro_f1/count_valid, valid_micro_f1/count_valid))
        epochs_end  =time.time()
        print("time : ",epochs_end - epoch_start)
        history.append([train_auc/count_item,train_macro_f1/count_item,train_micro_f1/count_item, valid_auc/count_valid, valid_macro_f1/count_valid, valid_micro_f1/count_valid])
        epoch_begin = 8 + rep
        torch.save(model, 'not_freeze_models/'+'resnet'+str(epoch+ 3 +epoch_begin)+'.pt')
    return model




if __name__ == "__main__":

    epochs = 1
    for i in range(3):
        model = torch.load("not_freeze_models/resnet{}.pt".format(10+i)).to(device)
        for param in model.parameters():
            print(param.requires_grad)
        print("not_freeze_models/resnet{}.pt".format(10+i))
        train_and_valid(model,loss_func,optimizer,epochs,train_data,valid_data,i)
   # print(resnet50)
    #print(fc_inputs)
