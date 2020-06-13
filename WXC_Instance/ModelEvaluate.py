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
from PIL import Image


use_gpu = torch.cuda.is_available()
print(use_gpu)
device = torch.device("cuda:2" if use_gpu else "cpu")

data_transforms = {
    'train': transforms.Compose([
        # 将图像进行缩放，缩放为256*256
        transforms.Resize(256),
        # 在256*256的图像上随机裁剪出227*227大小的图像用于训练
        transforms.RandomResizedCrop(227),
        # 图像用于翻转
        transforms.RandomHorizontalFlip(),
        # 转换成tensor向量
        transforms.ToTensor(),
        # 对图像进行归一化操作
        # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # 测试集需要中心裁剪，甚至不裁剪，直接缩放为224*224for，不需要翻转
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# 定义数据读入
def Load_Image_Information(path):
    # 图像存储路径
    image_Root_Dir = r''

    # 获取图像的路径
    iamge_Dir = os.path.join(image_Root_Dir, path)
    # 以RGB格式打开图像
    # Pytorch DataLoader就是使用PIL所读取的图像格式
    # 建议就用这种方法读取图像，当读入灰度图像时convert('')
    return Image.open(iamge_Dir).convert('RGB')


# 定义自己数据集的数据读入类
class my_Data_Set(nn.Module):
    def __init__(self, txt, transform=None, target_transform=None, loader=None):
        super(my_Data_Set, self).__init__()
        # 打开存储图像名与标签的txt文件
        fp = open(txt, 'r')
        images = []
        labels = []
        # 将图像名和图像标签对应存储起来
        for line in fp:
            line.strip('\n')
            line.rstrip()
            information = line.split()
            images.append(information[0])
            # 将标签信息由str类型转换为float类型
            labels.append([float(l) for l in information[1:len(information)]])
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    # 重写这个函数用来进行图像数据的读取
    def __getitem__(self, item):
        # 获取图像名和标签
        imageName = self.images[item]
        label = self.labels[item]
        # 读入图像信息
        image = self.loader(imageName)
        # 处理图像数据
        if self.transform is not None:
            image = self.transform(image)
        # 需要将标签转换为float类型，BCELoss只接受float类型
        label = torch.FloatTensor(label)
        return image, label

    # 重写这个函数，来看数据集中含有多少数据
    def __len__(self):
        return len(self.images)



model = models.resnet50(pretrained=False)
out_class = 10
model.fc = nn.Linear(model.fc.in_features, out_class)
model.load_state_dict(torch.load('./models/Resnet50_SI_9_epoch_model.pkl'))

# 定义损失函数
criterion = nn.BCELoss()

# model = model.cuda('2,3')
model = torch.nn.DataParallel(model, device_ids=[2, 3])
model = model.to(device)

val_Data = my_Data_Set(r'./index/tem.txt', transform=data_transforms['val'], loader=Load_Image_Information)
val_DataLoader = DataLoader(val_Data, batch_size=32)

valid_loss = 0.0
valid_auc = 0.0
valid_macro_f1 = 0.0
valid_micro_f1 = 0.0

count_item = 0
count_valid = 0
history = []

for valid_inputs, labels in val_DataLoader:
    valid_inputs = valid_inputs.transpose(1, 3).transpose(2, 3).to(device).float()
    labels = labels.to(device).float()
    outputs = model(valid_inputs)
    loss = criterion(outputs, labels)
    valid_loss += loss.item() * valid_inputs.size(0)

    res = torch.gt(outputs, 0.5)

    valid_auc += np.sum(np.array([evaluate.auc(labels[i].cpu(), res[i].cpu()) for i in range(res.size(0))]))
    valid_macro_f1 += np.sum(np.array([evaluate.macro_f1(labels[0].cpu(), res[0].cpu()) for i in range(res.size(0))]))
    valid_micro_f1 += np.sum(np.array([evaluate.micro_f1(labels[0].cpu(), res[0].cpu()) for i in range(res.size(0))]))
    count_valid += labels.size(0)
print("valid ---- epoch == > {}, auc ==> {},  macro_f1==> {}, micro_f1 ==> {}".format(0, valid_auc / count_valid,
                                                                                      valid_macro_f1 / count_valid,
                                                                                      valid_micro_f1 / count_valid))
epochs_end = time.time()
history.append(
    [valid_auc / count_valid, valid_macro_f1 / count_valid, valid_micro_f1 / count_valid])
