import os
import sys
import warnings
import numpy as np
from gensim.models import word2vec

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

def load_testing_data(path = 'testing_data.txt'):
    with open(path, 'r') as f:
        lines = f.readlines()
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
    return X

def train_word2vec(x):
    # training the word embedding of word to vector
    model = word2vec.Word2Vec(x, size = 250, window = 5, min_count = 5, workers = 12, iter = 10, sg = 1)
    return model

if __name__ == '__main__':
    input_file = sys.argv[1]

    # this is for filtering the warnings
    warnings.filterwarnings('ignore')

    print("loading training data ...")
    train_x, y = load_training_data(os.path.join(input_file, 'training_label.txt'))
    train_x_no_label = load_training_data(os.path.join(input_file, 'training_nolabel.txt'))

    print("loading testing data ...")
    test_x = load_testing_data(os.path.join(input_file, 'testing_data.txt'))

    num_label = 20000
    num_unlabel = 200000
    model = train_word2vec(train_x[:num_label] + train_x_no_label[:num_unlabel] + test_x)
    # model = train_word2vec(train_x + test_x)
    
    print("saving model ...")
    model.save('./model/w2v.model')