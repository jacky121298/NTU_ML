import sys
import torch
import random
import numpy as np
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.autograd import Variable
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from fcn import fcn_AE

if __name__ == '__main__':
    test_path = sys.argv[1]
    model_type = sys.argv[2]
    model_path = sys.argv[3]
    submit_csv = sys.argv[4]

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

    outlier = 3
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
    latents = list()
    for data in test_dataloader:
        if model_type == 'cnn':
            img = data[0].transpose(3, 1).cuda()
        else:
            img = data[0].cuda()
        output = model(img)
        
        latent = output[1].cpu().detach().numpy()
        latent = latent.reshape(len(latent), -1)
        latents.append(latent)

    latents = np.concatenate(latents, axis = 0)
    kmeans_x = MiniBatchKMeans(n_clusters = 5, random_state = 0).fit(latents)

    cluster = kmeans_x.predict(latents)
    anomality = np.sum(np.square(kmeans_x.cluster_centers_[cluster] - latents), axis = 1)
    anomality[cluster == outlier] += enhance
  
    with open(submit_csv, 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(anomality)):
            f.write('{},{}\n'.format(i + 1, anomality[i]))