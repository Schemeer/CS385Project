import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from DataSet import ImageDataset
import numpy as np
import os

class MIML(nn.Module):
    def __init__(self):
        super(MIML, self).__init__()
        basemodel = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(basemodel.children())[:-1])
        self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 10),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = torch.sum(x, 0)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    from sklearn import metrics
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    def auc(y_true,y_pred):
        auc = metrics.roc_auc_score(y_true,y_pred)
        return auc

    def macro_f1(y_true,y_pred):
        f1 = metrics.f1_score(y_true = y_true ,y_pred = y_pred, average = "macro")
        return f1

    def micro_f1(y_true,y_pred):
        f1 = metrics.f1_score(y_true = y_true ,y_pred = y_pred, average = "micro")
        return f1 

    epoch_num = 1
    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MIML().to(device)
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = ImageDataset(transform=transform)
    train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle = False)

    valid_data = ImageDataset(DataFile="Valid.pkl",transform=transform)
    valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle = False)
    train_loss = []
    for epoch in range(epoch_num):
        t_start = time.time()
        print(f"epoch: {epoch}/{epoch_num}")

        train_auc = 0.0
        train_macro_f1 = 0.0
        train_micro_f1 = 0.0

        valid_auc = 0.0
        valid_macro_f1 = 0.0
        valid_micro_f1 = 0.0

        count_item = 0
        count_valid = 0

        model.train()
        for batch_image, batch_label in tqdm(train_loader):
            batch_image = batch_image.view(-1,3,512,512).to(device)
            batch_label = batch_label.view(10).to(device)
            optimizer.zero_grad()

            output = model(batch_image)
            loss = loss_func(output, batch_label)
            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())

            res = torch.gt(output,0.5)

            train_auc += auc(batch_label.cpu(), res.cpu())
            train_macro_f1 += macro_f1(batch_label.cpu(), res.cpu())
            train_micro_f1 += micro_f1(batch_label.cpu(), res.cpu())
            count_item += 1
        print(f"train ---- epoch == > {epoch}, auc ==> {train_auc/count_item},  macro_f1==> {train_macro_f1/count_item}, micro_f1 ==> {train_micro_f1/count_item}")
        t_end  =time.time()
        print("time: ",t_end - t_start)

        model.eval()
        for batch_image, batch_label in tqdm(valid_loader):
            batch_image = batch_image.view(-1,3,512,512).to(device)
            batch_label = batch_label.view(10).to(device)
            optimizer.zero_grad()

            output = model(batch_image)

            loss = loss_func(output, batch_label)

            res = torch.gt(output,0.5)

            valid_auc += auc(batch_label.cpu(), res.cpu())
            valid_macro_f1 += macro_f1(batch_label.cpu(), res.cpu())
            valid_micro_f1 += micro_f1(batch_label.cpu(), res.cpu())
            count_valid += 1
        print(f"train ---- epoch == > {epoch}, auc ==> {valid_auc/count_valid},  macro_f1==> {valid_macro_f1/count_valid}, micro_f1 ==> {valid_micro_f1/count_valid}")
        t_end  =time.time()
        print("time: ",t_end - t_start)
        torch.save(model.state_dict(), f"models\\model_{epoch}.pth")
    plt.plot(train_loss)
    plt.savefig("result.png", dip=72)
    plt.show()