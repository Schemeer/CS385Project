import torch
from torchvision import models
import torch.nn as nn
from transformer import Transformer


class MIML(nn.Module):
    def __init__(self, group_size=8):
        super(MIML, self).__init__()
        basemodel = models.resnet18(pretrained=True)
        self.group_size = group_size
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
        x = x.view(-1, self.group_size, 512)
        x = torch.sum(x, 1)
        x = self.classifier(x)
        return x


class ImPlocMIML(nn.Module):
    def __init__(self, group_size=8):
        super(ImPlocMIML, self).__init__()
        self.group_size = group_size
        basemodel = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(basemodel.children())[:-1])
        self.Transformer = Transformer("res18-512")
        for p in self.feature_extractor.parameters():
            p.require_grad = False
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = x.view(-1, 8, 512)
        x = self.Transformer(x)
        return x

class FineTuneModel(torch.nn.Module):

    def __init__(self, original_model):
        super(FineTuneModel, self).__init__()

        self.features = torch.nn.Sequential(
                        *list(original_model.children())[:-2])

        for p in self.features.parameters():
            p.require_grad = False

    def forward(self, x):
        f = self.features(x)
        f = torch.nn.AdaptiveAvgPool2d(1)(f)
        return f.view(f.size(0), -1)

if __name__=="__main__":
    original_model = models.resnet18(pretrained=False)
    model = torch.nn.Sequential(*list(original_model.children())[:-4])
    print(model)