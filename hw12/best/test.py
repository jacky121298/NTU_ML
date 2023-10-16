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
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.autograd import Function

from FeatureExtractor import FeatureExtractor
from LabelPredictor import LabelPredictor
from DomainClassifier import DomainClassifier

if __name__ == '__main__':
    data_path = sys.argv[1]
    extractor_model = sys.argv[2]
    predictor_model = sys.argv[3]
    submit_csv = sys.argv[4]
    
    # fix the random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

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

    target_dataset = ImageFolder(os.path.join(data_path, 'test_data'), transform = target_transform)
    test_dataloader = DataLoader(target_dataset, batch_size = 128, shuffle = False)

    feature_extractor = FeatureExtractor().cuda()
    label_predictor = LabelPredictor().cuda()
    
    feature_extractor.load_state_dict(torch.load(extractor_model))
    label_predictor.load_state_dict(torch.load(predictor_model))

    result = []
    feature_extractor.eval()
    label_predictor.eval()
    
    for i, (test_data, _) in enumerate(test_dataloader):
        test_data = test_data.cuda()
        class_logits = label_predictor(feature_extractor(test_data))

        x = torch.argmax(class_logits, dim = 1).cpu().detach().numpy()
        result.append(x)

    # Generate your submission
    result = np.concatenate(result)
    df = pd.DataFrame({'id': np.arange(len(result)), 'label': result})
    df.to_csv(submit_csv, index = False)