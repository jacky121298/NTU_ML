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
    input_file = sys.argv[2]
    submit_csv = sys.argv[3]

    print("Reading data...")
    test_x = readfile(os.path.join(workspace_dir, "testing"), False)
    print("Size of Testing data = {}".format(len(test_x)))

    model_best = Classifier().cuda()
    model_best.load_state_dict(torch.load(input_file))

    # testing : no data augmentation
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    batch_size = 32
    test_set = ImgDataset(test_x, transform = test_transform)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)

    model_best.eval()
    prediction = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = model_best(data.cuda())
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            for y in test_label:
                prediction.append(y)

    with open(submit_csv, 'w') as f:
        f.write('Id,Category\n')
        for i, y in enumerate(prediction):
            f.write('{},{}\n'.format(i, y))
