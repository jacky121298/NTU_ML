import os
import sys
import torch
import pandas as pd
from torch import nn
from torch.utils import data
from preprocess import Preprocess

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
        self.classifier = nn.Sequential( nn.Dropout(dropout), nn.Linear(hidden_dim, 1), nn.Sigmoid() )
    
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最後一層的 hidden state
        x = x[:, -1, :] 
        x = self.classifier(x)
        return x

def load_testing_data(path = 'testing_data.txt'):
    with open(path, 'r') as f:
        lines = f.readlines()
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
    return X

def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype = torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0
            ret_output += outputs.int().tolist()
    
    return ret_output

if __name__ == '__main__':
    test_file = sys.argv[1]
    w2v_path = sys.argv[2]
    model_path = sys.argv[3]
    output_file = sys.argv[4]

    sen_len = 32
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("loading testing data ...")
    testing_data = os.path.join(test_file, 'testing_data.txt')
    test_x = load_testing_data(testing_data)
    
    preprocess = Preprocess(test_x, sen_len, w2v_path = w2v_path)
    embedding = preprocess.make_embedding(load = True)
    test_x = preprocess.sentence_word2idx()
    test_dataset = TwitterDataset(X = test_x, y = None)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)
    
    print("\nload model ...")
    model = torch.load(model_path)
    outputs = testing(batch_size, test_loader, model, device)

    tmp = pd.DataFrame({'id' : [str(i) for i in range(len(test_x))], 'label' : outputs})
    print("save csv ...")
    tmp.to_csv(output_file, index = False)