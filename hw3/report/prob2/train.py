import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import random

def readfile(path, label):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype = np.uint8)
    y = np.zeros((len(image_dir)), dtype = np.uint8)

    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img, (128, 128))
        if label: y[i] = int(file.split("_")[0])
    
    if label: return x, y
    else: return x

class ImgDataset(Dataset):
    def __init__(self, x, y = None, transform = None):
        self.x, self.y, self.transform = x, y, transform
        
        # label is required to be a LongTensor
        if y is not None:
            self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else: return X

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input dimension [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1),  # [128, 128, 128]
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 64, 64]

            nn.Conv2d(128, 512, 3, 1, 1), # [512, 64, 64]
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.MaxPool2d(4, 4, 0),      # [512, 16, 16]

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.MaxPool2d(4, 4, 0),       # [512, 4, 4]
        )

        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.Dropout(p = 0.5, inplace = True),
            nn.PReLU(),

            nn.Linear(1024, 512),
            nn.Dropout(p = 0.5, inplace = True),
            nn.PReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

if __name__ == '__main__':
    workspace_dir = sys.argv[1]
    output_file = sys.argv[2]

    print("Reading data...")
    train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
    print("Size of training data = {}".format(len(train_x)))
    val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
    print("Size of validation data = {}".format(len(val_x)))

    # fix the random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    
    # training : data augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # 15 degrees
        transforms.ToTensor(), # normalize data to [0, 1] (data normalization)
    ])
    # testing : no data augmentation
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    batch_size = 32
    train_set = ImgDataset(train_x, train_y, train_transform)
    val_set = ImgDataset(val_x, val_y, test_transform)
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False)

    model = Classifier().cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    num_epoch = 200

    for epoch in range(num_epoch):
        start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()                                      ## 確保 model 是在 train mode (開啟 Dropout 等...)
        for i, data in enumerate(train_loader):            ##
            optimizer.zero_grad()                          ## 用 optimizer 將 model 參數的 gradient 歸零
            train_pred = model(data[0].cuda())             ## 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
            batch_loss = loss(train_pred, data[1].cuda())  ## 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
            batch_loss.backward()                          ## 利用 back propagation 算出每個參數的 gradient
            optimizer.step()                               ## 以 optimizer 用 gradient 更新參數值

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis = 1) == data[1].numpy())
            train_loss += batch_loss.item()
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred = model(data[0].cuda())
                batch_loss = loss(val_pred, data[1].cuda())

                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis = 1) == data[1].numpy())
                val_loss += batch_loss.item()

            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % (epoch + 1, num_epoch, time.time() - start_time, \
                train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))

    train_val_x = np.concatenate((train_x, val_x), axis = 0)
    train_val_y = np.concatenate((train_y, val_y), axis = 0)
    train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
    train_val_loader = DataLoader(train_val_set, batch_size = batch_size, shuffle = True)

    model_best = Classifier().cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_best.parameters(), lr = 0.001)
    num_epoch = 200

    for epoch in range(num_epoch):
        start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0

        model_best.train()
        for i, data in enumerate(train_val_loader):
            optimizer.zero_grad()
            train_pred = model_best(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda())
            batch_loss.backward()
            optimizer.step()

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis = 1) == data[1].numpy())
            train_loss += batch_loss.item()

        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % (epoch + 1, num_epoch, time.time() - start_time, \
            train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))

    torch.save(model_best.state_dict(), output_file)
