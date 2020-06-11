
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
resnet50 = models.resnet50(pretrained = True)
for param in resnet50.parameters():
    param.requires_grad = False

fc_inputs = 8192
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 10),
    nn.Sigmoid()
)


resnet50 = resnet50.to('cuda:0')
loss_func = nn.BCELoss()
optimizer = optim.Adam(resnet50.parameters())


def train_and_valid(model, loss_function,optimizer,epochs,train_data,valid_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1,epochs))
        model.train()
        train_loss = 0.0
        train_auc = 0.0
        train_macro_f1 = 0.0
        train_micro_f1 = 0.0

        valid_loss = 0.0
        valid_acc = 0.0
        valid_macro_f1 = 0.0
        valid_micro_f1 = 0.0

        count_item = 0
        count_valid = 0

        for inputs,labels in train_data:
            print("-----train mode-----")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item() * inputs.size(0)

            res = torch.gt(outputs,0.5)
            
            train_auc += np.sum(np.array([evaluate.auc(labels[i].cpu(),res[i].cpu()) for i in range(res.size(0))]))
            train_macro_f1 = np.sum(np.array([evaluate.macro_f1(labels[0].cpu(),res[0].cpu())  for i in range(res.size(0))]))
            train_micro_f1 = np.sum(np.array([evaluate.micro_f1(labels[0].cpu(),res[0].cpu())  for i in range(res.size(0))]))
            count_item += labels.size(0)
        print("train ---- epoch == > {}, auc ==> {},  macro_f1==> {}, micro_f1 ==> {}".format(epoch,auc/count_item,macro_f1/count_item,micro_f1/count_item))
        epochs_end  =time.time()
        print("time : ",epochs_end - epoch_start)

        with torch.no_grad():
            model.eval()
        for valid_inputs,labels in valid_data:
            outputs = model(valid_inputs)
            loss = loss_function(outputs, labels)
            valid_loss += loss.item() * valid_inputs.size(0)

            res = torch.gt(outputs,0.5)
        
            auc = np.sum(np.array([evaluate.auc(labels[i].cpu(),res[i].cpu()) for i in range(res.size(0))]))
            macro_f1 = np.sum(np.array([evaluate.macro_f1(labels[0].cpu(),res[0].cpu())  for i in range(res.size(0))]))
            micro_f1 = np.sum(np.array([evaluate.micro_f1(labels[0].cpu(),res[0].cpu())  for i in range(res.size(0))]))
            count_valid += labels.size(0)
        print("valid ---- epoch == > {}, auc ==> {},  macro_f1==> {}, micro_f1 ==> {}".format(epoch,auc/count_valid,macro_f1/count_valid,micro_f1/count_valid))
        epochs_end  =time.time()
        print("time : ",epochs_end - epoch_start)




if __name__ == "__main__":

    epochs = 1
    train_and_valid(resnet50,loss_func,optimizer,epochs,train_data,valid_data)
   # print(resnet50)
    #print(fc_inputs)
