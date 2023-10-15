import os
import sys
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.autograd import Function

from FeatureExtractor import FeatureExtractor
from LabelPredictor import LabelPredictor

def train_epoch(source_dataloader):
    running_F_loss = 0.0
    total_hit, total_num = 0.0, 0.0

    for i, (source_data, source_label) in enumerate(source_dataloader):
        source_data = source_data.cuda()
        source_label = source_label.cuda()

        feature = feature_extractor(source_data)
        class_logits = label_predictor(feature)
        loss = class_criterion(class_logits, source_label)
        
        running_F_loss += loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim = 1) == source_label).item()
        total_num += source_data.shape[0]
        print('batch # ', i, end = '\r')

    return running_F_loss / (i+1), total_hit / total_num

if __name__ == '__main__':
    data_path = sys.argv[1]
    extractor_model = sys.argv[2]

    # fix the random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    source_transform = transforms.Compose([
        # 轉灰階: Canny 不吃 RGB
        transforms.Grayscale(),
        # cv2 不吃 skimage.Image，因此轉成np.array後再做cv2.Canny
        transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
        # 重新將np.array 轉回 skimage.Image
        transforms.ToPILImage(),
        # 水平翻轉 (Augmentation)
        transforms.RandomHorizontalFlip(),
        # 旋轉15度內 (Augmentation)，旋轉後空的地方補 0
        transforms.RandomRotation(15, fill = (0,)),
        # 最後轉成Tensor供model使用
        transforms.ToTensor(),
    ])

    source_dataset = ImageFolder(os.path.join(data_path, 'train_data'), transform = source_transform)
    source_dataloader = DataLoader(source_dataset, batch_size = 32, shuffle = True)

    feature_extractor = FeatureExtractor().cuda()
    label_predictor = LabelPredictor().cuda()

    class_criterion = nn.CrossEntropyLoss()
    optimizer_F = optim.Adam(feature_extractor.parameters())
    optimizer_C = optim.Adam(label_predictor.parameters())

    nepoch = 1000
    for epoch in range(nepoch):
        train_F_loss, train_acc = train_epoch(source_dataloader)
        torch.save(feature_extractor.state_dict(), extractor_model)

        print('epoch {:>3d} | train F loss: {:6.4f} | acc {:6.4f}'.format(epoch + 1, train_F_loss, train_acc))