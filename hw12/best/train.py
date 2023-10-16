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
from DomainClassifier import DomainClassifier

def train_epoch(source_dataloader, target_dataloader, lamb):
    '''
      Args:
        source_dataloader: source data's dataloader
        target_dataloader: target data's dataloader
        lamb: 調控 adversarial 的 loss parameter
    '''
    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    # total_hit: 計算目前對了幾筆 total_num: 目前經過了幾筆
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0

    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = target_data.cuda()
        
        # 我們把source data和target data混在一起，否則batch_norm可能會算錯 (兩邊的data的mean/var不太一樣)
        mixed_data = torch.cat([source_data, target_data], dim = 0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        # 設定source data的label為1
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : 訓練Domain Classifier
        feature = feature_extractor(mixed_data)
        # 因為我們在Step 1不需要訓練Feature Extractor，所以把feature detach避免loss backprop上去
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss += loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2 : 訓練Feature Extractor和Domain Classifier
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss為原本的class CE - lamb * domain BCE，相減的原因同GAN中的Discriminator中的G loss
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss += loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim = 1) == source_label).item()
        total_num += source_data.shape[0]
        print('batch # ', i, end = '\r')

    return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num

if __name__ == '__main__':
    data_path = sys.argv[1]
    extractor_model = sys.argv[2]
    predictor_model = sys.argv[3]

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

    source_dataloader = DataLoader(source_dataset, batch_size = 32, shuffle = True)
    target_dataloader = DataLoader(target_dataset, batch_size = 32, shuffle = True)

    feature_extractor = FeatureExtractor().cuda()
    label_predictor = LabelPredictor().cuda()
    domain_classifier = DomainClassifier().cuda()

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer_F = optim.Adam(feature_extractor.parameters())
    optimizer_C = optim.Adam(label_predictor.parameters())
    optimizer_D = optim.Adam(domain_classifier.parameters())

    nepoch = 2000
    for epoch in range(nepoch):
        # adaptive parameter lambda
        p = epoch / nepoch
        lamb = (2 / (1 + np.exp(-10 * p))) - 1
        train_D_loss, train_F_loss, train_acc = train_epoch(source_dataloader, target_dataloader, lamb = lamb)

        torch.save(feature_extractor.state_dict(), extractor_model)
        torch.save(label_predictor.state_dict(), predictor_model)

        print('epoch {:>3d} | train D loss: {:6.4f}, train F loss: {:6.4f} | acc {:6.4f}'.format(epoch + 1, train_D_loss, train_F_loss, train_acc))