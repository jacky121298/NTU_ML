import os
import sys
import cv2
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.autograd import Function

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from FeatureExtractor import FeatureExtractor

def get_feature(dataloader, size):
    features = []
    
    for i, (data, _) in enumerate(dataloader):
        if i >= size: break
        data = data.cuda()
        feature = feature_extractor(data)
        features.append(feature.cpu().detach().numpy().reshape(1, -1))

    features = np.concatenate(features)
    return features

if __name__ == '__main__':
    data_path = sys.argv[1]
    extractor_model = sys.argv[2]

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

    target_transform = transforms.Compose([
        # 轉灰階: 將輸入3維壓成1維
        transforms.Grayscale(),
        # 縮放: 因為source data是 (32 x 32)，我們將target data的(28 x 28)放大成(32 x 32)
        transforms.Resize((32, 32)),
        # 水平翻轉 (Augmentation)
        transforms.RandomHorizontalFlip(),
        # 旋轉15度內 (Augmentation)，旋轉後空的地方補 0
        transforms.RandomRotation(15, fill = (0,)),
        # 最後轉成Tensor供model使用
        transforms.ToTensor(),
    ])

    source_dataset = ImageFolder(os.path.join(data_path, 'train_data'), transform = source_transform)
    target_dataset = ImageFolder(os.path.join(data_path, 'test_data'), transform = target_transform)
    
    source_dataloader = DataLoader(source_dataset, batch_size = 1, shuffle = True)
    target_dataloader = DataLoader(target_dataset, batch_size = 1, shuffle = True)

    feature_extractor = FeatureExtractor().cuda()
    feature_extractor.load_state_dict(torch.load(extractor_model))
    feature_extractor.eval()

    source_feature = get_feature(source_dataloader, size = 5000)
    target_feature = get_feature(target_dataloader, size = 5000)

    print('dimension reduction')
    source_tsne = TSNE(n_components = 2).fit_transform(source_feature)
    target_tsne = TSNE(n_components = 2).fit_transform(target_feature)
    print('source shape:', source_tsne.shape)
    print('target shape:', target_tsne.shape)

    plt.figure()
    plt.scatter(source_tsne[:, 0], source_tsne[:, 1], marker = '.', c = 'r')
    plt.scatter(target_tsne[:, 0], target_tsne[:, 1], marker = '.', c = 'b')

    plt.title('DaNN')
    plt.legend(['source', 'target'])

    plt.tight_layout()
    plt.savefig('prob3.png')