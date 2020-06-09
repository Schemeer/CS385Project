
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import DataPrecess
import numpy as np
import os


batch_size = 32
num_classes = 10
train_filename = 'tr.txt'
train_data = DataPrecess.ImageDataset(train_filename)
train_data = DataLoader(dataset = train_data, batch_size=batch_size, shuffle = False)


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
# loss_func = nn.BCEloss()
# loss_func = nn.CrossEntropy()
loss_func = nn.BCELoss()
optimizer = optim.Adam(resnet50.parameters())
epoch = 25


def train_and_valid(model, loss_function,optimizer,epochs,train_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1,epochs))
    
        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        # for batch_image, batch_label in train_data:
        #     print(batch_image.shape)
        #     print(batch_label.shape)

        
        inputs= torch.rand(32,3,512,512).to(device)
        labels = torch.randint(0,2,(32,10),dtype = torch.float).to(device)
        # print(labels)
        # print(inputs.shape)
        # print(labels.shape)
      
        optimizer.zero_grad()
        outputs = model(inputs)
        #print(outputs)
   
        loss = loss_function(outputs, labels)
        
        loss.backward()
        optimizer.step()
     
        train_loss += loss.item() * inputs.size(0)
 
        res = torch.gt(outputs,0.5)
        countTrue = torch.eq(res,labels).sum().item()
        TotalCount = labels.size(0)*labels.size(1)

        acc = countTrue/TotalCount
        print("acc ==>{}".format(str(acc)))

        epochs_end  =time.time()

        
        
        print("time : ",epochs_end - epoch_start)
if __name__ == "__main__":

    epochs = 1
    train_and_valid(resnet50,loss_func,optimizer,epochs,train_data)
   # print(resnet50)
    #print(fc_inputs)
