import torch
from torch import nn

class Preprocess():
    def __init__(self, sentences):
        self.sentences = sentences
        self.idx2word = []
        self.word2idx = {}
    
    def BOW(self):
        for i, sentence in enumerate(self.sentences):
            for word in sentence:
                if word not in self.word2idx.keys():
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word.append(word)
        
        print("total words: {}".format(len(self.word2idx)))
        return
    
    def BOW_vector(self, inputs):
        vector = torch.zeros(len(inputs), len(self.word2idx))
        for i, sentence in enumerate(inputs):
            for word in sentence:
                vector[i][self.word2idx[word]] += 1        
        return vector
    
    def labels_to_tensor(self, y):
        # 把 labels 轉成 tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)

    def vector_dim(self):
        return len(self.word2idx)