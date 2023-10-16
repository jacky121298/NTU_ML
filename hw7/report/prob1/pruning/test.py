import os
import sys
import re
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from glob import glob
from PIL import Image
from model import StudentNet

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, folderName, transform = None):
        self.transform = transform
        self.data = []

        for img_path in sorted(glob(folderName + '/*.jpg')):
            image = Image.open(img_path)
            image_fp = image.fp
            image.load()
            image_fp.close()
            self.data.append(image)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image

def get_dataloader(mode = 'testing', batch_size = 32, data_dir = './food-11'):
    assert mode in ['training', 'testing', 'validation']
    dataset = MyDataset(os.path.join(data_dir, mode), transform = trainTransform if mode == 'training' else testTransform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = (mode == 'training'))
    return dataloader

if __name__ == '__main__':
    data_dir = sys.argv[1]
    model_path = sys.argv[2]
    width_mult = float(sys.argv[3])
    submit_csv = sys.argv[4]

    trainTransform = transforms.Compose([
        transforms.RandomCrop(256, pad_if_needed = True, padding_mode = 'symmetric'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    testTransform = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    batch_size = 32
    print('reading testing data ...')
    testing_dataloader = get_dataloader('testing', batch_size = batch_size, data_dir = data_dir)

    model = StudentNet(base = 16, width_mult = width_mult).cuda()
    model.load_state_dict(torch.load(model_path))

    model.eval()
    prediction = []
    
    with torch.no_grad():
        for i, data in enumerate(testing_dataloader):
            test_pred = model(data.cuda())
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis = 1)
            for y in test_label:
                prediction.append(y)

    with open(submit_csv, 'w') as f:
        f.write('Id,Label\n')
        for i, y in enumerate(prediction):
            f.write('{},{}\n'.format(i, y))