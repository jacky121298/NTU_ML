import os
import sys
import torch
import numpy as np
from torch import nn
from torch.utils import data
from gensim.models import word2vec
from preprocess import Preprocess
from training import training
import random

class TwitterDataset(data.Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)
    
    __len__ will return the number of data
    """
    def __init__(self, X, y):
        self.data = X
        self.label = y
    
    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]
    
    def __len__(self):
        return len(self.data)

class BOW(nn.Module):
    def __init__(self, vector_dim, dropout = 0.5):
        super(BOW, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(vector_dim, 1024),
            nn.PReLU(),

            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.PReLU(),

            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.PReLU(),

            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.classifier(inputs)
        return x

def load_training_data(path = 'training_label.txt'):
    if 'training_label' in path:
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]      
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        return x, y
    
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x

if __name__ == '__main__':
    label_data = sys.argv[1]
    unlabel_data = sys.argv[2]

    # fix the random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    train_with_label = os.path.join(label_data, 'training_label.txt')
    train_no_label = os.path.join(unlabel_data, 'training_nolabel.txt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    epoch = 10
    lr = 0.001
    model_dir = './' # model directory for checkpoint model

    print("loading data ...") # 把 'training_label.txt' 跟 'training_nolabel.txt' 讀進來
    train_x, y = load_training_data(train_with_label)
    train_x_no_label = load_training_data(train_no_label)
    test_x = ['today is a good day , but it is hot', 'today is hot , but it is a good day']
    test_x = [sen.split() for sen in test_x]

    preprocess = Preprocess(train_x + test_x)
    preprocess.BOW()

    # 把 data 分為 training data 跟 validation data（將一部份 training data 拿去當作 validation data）
    X_train, X_val, y_train, y_val = train_x[:180000], train_x[180000:], y[:180000], y[180000:]

    # 製作一個 model 的對象
    model = BOW(vector_dim = preprocess.vector_dim(), dropout = 0.5)
    model = model.to(device) # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）

    training(batch_size, epoch, lr, model_dir, X_train, X_val, y_train, y_val, model, device, preprocess)