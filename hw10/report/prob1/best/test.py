import sys
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam, AdamW
from torch.autograd import Variable
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from fcn import fcn_AE

if __name__ == '__main__':
    test_path = sys.argv[1]
    model_type = sys.argv[2]
    model_path = sys.argv[3]

    # fix the random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    
    print('load testing data ...')
    test = np.load(test_path, allow_pickle = True)
    if model_type == 'fcn' or model_type == 'vae':
        test = test.reshape(len(test), -1)

    outlier = 1
    enhance = 5000
    batch_size = 128
    data = torch.tensor(test, dtype = torch.float)

    models = {'fcn' : fcn_AE()}
    model = models[model_type].cuda()
    model.load_state_dict(torch.load(model_path))
    
    test_dataset = TensorDataset(data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler = test_sampler, batch_size = batch_size)
    
    model.eval()
    reconstructs = list()
    for data in test_dataloader: 
        if model_type == 'cnn':
            img = data[0].transpose(3, 1).cuda()
        else:
            img = data[0].cuda()
        output = model(img)
        
        reconstruct = output[0].cpu().detach().numpy()
        reconstruct = reconstruct.reshape(len(reconstruct), -1)
        reconstructs.append(reconstruct)

    reconstructs = np.concatenate(reconstructs, axis = 0)
    kmeans_x = MiniBatchKMeans(n_clusters = 5, random_state = 0).fit(reconstructs)

    cluster = kmeans_x.predict(reconstructs)
    anomality = np.sum(np.square(kmeans_x.cluster_centers_[cluster] - reconstructs), axis = 1)
    anomality[cluster == outlier] += enhance
  
    mse = np.sqrt(np.sum(np.square(reconstructs - test).reshape(len(test), -1), axis = 1))
    mse = np.argsort(mse, axis = 0)
    target = np.concatenate((mse[:2], mse[-2:]))

    fig, axs = plt.subplots(2, len(target))
    for i, idx in enumerate(target):
        axs[0][i].imshow(test[idx].reshape(32, 32, 3))
        axs[1][i].imshow(reconstructs[idx].reshape(32, 32, 3))

    plt.tight_layout()
    plt.savefig('best.png')
    plt.close()