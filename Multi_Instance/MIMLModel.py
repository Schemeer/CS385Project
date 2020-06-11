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
        basemodel = models.resnet18(pretrained=False)
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
    pass
    # epoch_num = 2
    # batch_size = 1
    # transform = transforms.Compose([transforms.ToTensor()])
    # train_data = ImageDataset(transform=transform)
    # train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle = False)
    # model = MIML()
    # for epoch in range(epoch_num):
    #     for batch_image, batch_label in train_loader:
    #         batch_image = batch_image.view(-1,3,512,512)
    #         batch_label = batch_label.view(10)
    #         res = model(batch_image)