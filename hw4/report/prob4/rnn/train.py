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

class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout = 0.5, fix_embedding = True):
        super(LSTM_Net, self).__init__()
        # 製作 embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # 是否將 embedding fix 住，如果 fix_embedding 為 False，在訓練過程中，embedding 也會跟著被訓練
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = num_layers, batch_first = True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最後一層的 hidden state
        x = x[:, -1, :] 
        x = self.classifier(x)
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
    w2v_path = sys.argv[3]

    # fix the random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    train_with_label = os.path.join(label_data, 'training_label.txt')
    train_no_label = os.path.join(unlabel_data, 'training_nolabel.txt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sen_len = 32
    fix_embedding = True # fix embedding during training
    batch_size = 128
    epoch = 10
    lr = 0.001
    model_dir = './model' # model directory for checkpoint model

    print("loading data ...") # 把 'training_label.txt' 跟 'training_nolabel.txt' 讀進來
    train_x, y = load_training_data(train_with_label)
    train_x_no_label = load_training_data(train_no_label)

    num_label = 20000
    # 對 input 跟 labels 做預處理
    preprocess = Preprocess(train_x[:num_label], sen_len, w2v_path = w2v_path)
    embedding = preprocess.make_embedding(load = True)
    train_x = preprocess.sentence_word2idx()
    y = preprocess.labels_to_tensor(y)

    # 製作一個 model 的對象
    model = LSTM_Net(embedding, embedding_dim = 250, hidden_dim = 150, num_layers = 1, dropout = 0.5, fix_embedding = fix_embedding)
    model = model.to(device) # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）

    # 把 data 分為 training data 跟 validation data（將一部份 training data 拿去當作 validation data）
    X_train, X_val, y_train, y_val = train_x[:18000], train_x[18000:], y[:18000], y[18000:]

    # 把 data 做成 dataset 供 dataloader 取用
    train_dataset = TwitterDataset(X = X_train, y = y_train)
    val_dataset = TwitterDataset(X = X_val, y = y_val)

    # 把 data 轉成 batch of tensors
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = 8)
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)

    training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)