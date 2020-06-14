import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from DataSet import ImageDataset, KfoldDataset
from MIMLModel import MIML, ImPlocMIML
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import numpy as np
import os



def auc(y_true,y_pred):
    auc = metrics.roc_auc_score(y_true,y_pred)
    return auc

def macro_f1(y_true,y_pred):
    f1 = metrics.f1_score(y_true = y_true ,y_pred = y_pred, average = "macro")
    return f1

def micro_f1(y_true,y_pred):
    f1 = metrics.f1_score(y_true = y_true ,y_pred = y_pred, average = "micro")
    return f1 

def threshold_tensor_batch(pred, base=0.5):
    p_max = torch.max(pred, dim=1)[0]
    pivot = torch.cuda.FloatTensor([base]).expand_as(p_max)
    threshold = torch.min(0.9*p_max, pivot)
    pred_label = torch.ge(pred, threshold.unsqueeze(dim=1))
    return pred_label


if __name__ == "__main__":


    epoch_num = 5
    batch_size = 2
    group_size = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ImPlocMIML(group_size=group_size)
    # model.load_state_dict(torch.load("Multi_Instance\\models\\model_2.pth"))
    model.to(device)
    loss_func = nn.BCELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.0005)

    transform = transforms.Compose([transforms.ToTensor()])
    train_loss = []
    all_fold = [0,1,2,3,4]
    for epoch in range(epoch_num):
        train_fold = [i for i in all_fold if i!=epoch]
        train_data = KfoldDataset(transform=transform, FoldIds=train_fold)
        train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle = True)

        valid_fold = [epoch]
        valid_data = KfoldDataset(transform=transform, FoldIds=valid_fold)
        valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle = False)
        t_start = time.time()
        print(f"epoch: {epoch}/{epoch_num}, train folds: {train_fold}, valid_folds: {valid_fold}")
        Train_pred = []
        Train_GT = []
        model.train()
        for batch_image, batch_label in tqdm(train_loader):
            batch_image = batch_image.view(-1,3,512,512).to(device)
            batch_label = batch_label.to(device)
            optimizer.zero_grad()

            output = model(batch_image)
            all_loss = loss_func(output, batch_label)
            label_loss = torch.sum(all_loss, dim=0)
            loss = torch.sum(label_loss)
            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())
            res = threshold_tensor_batch(output, 0.5)
            Train_pred.append(res.cpu().numpy())
            Train_GT.append(batch_label.cpu().numpy())
        Train_pred = np.concatenate(Train_pred)
        Train_GT = np.concatenate(Train_GT)
        print(Train_pred[-10:])
        print(Train_GT[-10:])
        print(f"train ---- epoch == > {epoch}, auc ==> {auc(Train_GT, Train_pred)},  macro_f1==> {macro_f1(Train_GT, Train_pred)}, micro_f1 ==> {micro_f1(Train_GT, Train_pred)}")
        t_end  = time.time()
        print("time: ",t_end - t_start)

        Valid_pred = []
        Valid_GT = []
        model.eval()
        for batch_image, batch_label in tqdm(valid_loader):
            batch_image = batch_image.view(-1,3,512,512).to(device)
            batch_label = batch_label

            output = model(batch_image)

            res = threshold_tensor_batch(output, 0.5)

            Valid_pred.append(res.cpu().numpy())
            Valid_GT.append(batch_label.numpy())

        Valid_pred = np.concatenate(Valid_pred)
        Valid_GT = np.concatenate(Valid_GT)
        print(Valid_pred[-10:])
        print(Valid_GT[-10:])
        print(f"valid ---- epoch == > {epoch}, auc ==> {auc(Valid_GT, Valid_pred)},  macro_f1==> {macro_f1(Valid_GT, Valid_pred)}, micro_f1 ==> {micro_f1(Valid_GT, Valid_pred)}")
        t_end  = time.time()
        print("time: ",t_end - t_start)
        torch.save(model.state_dict(), f"Multi_Instance\\models\\model_{epoch}.pth")
    plt.plot(train_loss)
    plt.savefig("result.png", dip=72)
    plt.show()