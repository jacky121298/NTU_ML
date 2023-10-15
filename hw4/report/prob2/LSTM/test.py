import os
import sys
import torch
import pandas as pd
from torch import nn
from preprocess import Preprocess

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

if __name__ == '__main__':
    w2v_path = sys.argv[1]
    model_path = sys.argv[2]

    sen_len = 32
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test = ['today is a good day , but it is hot', 'today is hot , but it is a good day']
    test_x = [sen.split() for sen in test]

    preprocess = Preprocess(test_x, sen_len, w2v_path = w2v_path)
    embedding = preprocess.make_embedding(load = True)
    test_x = preprocess.sentence_word2idx()
    
    print("\nload model ...")
    model = torch.load(model_path)

    model.eval()
    with torch.no_grad():
        test_x = test_x.to(device, dtype = torch.long)
        outputs = model(test_x)
        outputs = outputs.squeeze()
        
        print('\nIn LSTM model:')
        print(test[0] + ':', outputs[0].item())
        print(test[1] + ':', outputs[1].item())