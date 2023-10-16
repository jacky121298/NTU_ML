import sys
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from fcn import fcn_AE
from conv import conv_AE
from vae import VAE

if __name__ == '__main__':
    test_path = sys.argv[1]
    model_type = sys.argv[2]
    model_path = sys.argv[3]
    submit_csv = sys.argv[4]
    
    print('load testing data ...')
    test = np.load(test_path, allow_pickle = True)
    if model_type == 'fcn' or model_type == 'vae':
        test = test.reshape(len(test), -1)

    batch_size = 128
    data = torch.tensor(test, dtype = torch.float)

    models = {'fcn' : fcn_AE(), 'cnn' : conv_AE(), 'vae' : VAE()}
    model = models[model_type].cuda()
    model.load_state_dict(torch.load(model_path))
    
    test_dataset = TensorDataset(data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler = test_sampler, batch_size = batch_size)
    
    model.eval()
    reconstructed = list()
    for data in test_dataloader: 
        if model_type == 'cnn':
            img = data[0].transpose(3, 1).cuda()
        else:
            img = data[0].cuda()
        output = model(img)
        
        if model_type == 'cnn':
            output = output.transpose(3, 1)
        elif model_type == 'vae':
            output = output[0]
        reconstructed.append(output.cpu().detach().numpy())

    reconstructed = np.concatenate(reconstructed, axis = 0)
    anomality = np.sqrt(np.sum(np.square(reconstructed - test).reshape(len(test), -1), axis = 1))
    
    with open(submit_csv, 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(anomality)):
            f.write('{},{}\n'.format(i + 1, anomality[i]))