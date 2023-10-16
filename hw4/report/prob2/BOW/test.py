import os
import sys
import torch
import pandas as pd
from torch import nn
from preprocess import Preprocess

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
    model_path = sys.argv[1]
    label_data = sys.argv[2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test = ['today is a good day , but it is hot', 'today is hot , but it is a good day']
    test_x = [sen.split() for sen in test]
    train_with_label = os.path.join(label_data, 'training_label.txt')

    print("load bow.model ...")
    model = torch.load(model_path)

    train_x, _ = load_training_data(train_with_label)
    preprocess = Preprocess(train_x + test_x)
    preprocess.BOW()

    model.eval()
    with torch.no_grad():
        inputs = preprocess.BOW_vector(test_x)
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = outputs.squeeze()
        
        print('\nIn BOW model:')
        print(test[0] + ':', outputs[0].item())
        print(test[1] + ':', outputs[1].item())