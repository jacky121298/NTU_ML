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
        self.fc = nn.Sequential(
            nn.Linear(3*128*128, 256),
            nn.PReLU(),

            nn.Linear(256, 512),
            nn.PReLU(),

            nn.Linear(512, 1024),
            nn.PReLU(),

            nn.Linear(1024, 2048),
            nn.PReLU(),
            nn.Linear(2048, 11)
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return self.fc(x)

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

    batch_size = 64
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
