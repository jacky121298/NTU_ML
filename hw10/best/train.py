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

def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x : generating images
    x : origin images
    mu : latent mean
    logvar : latent log variance
    """
    mse = criterion(recon_x, x)
    # KL Divergence
    # loss = 0.5 * sum(1 + log(sigma ^ 2) - mu ^ 2 - sigma ^ 2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mse + KLD

if __name__ == '__main__':
    train_path = sys.argv[1]
    model_type = sys.argv[2]
    model_path = sys.argv[3]

    print('load training data ...')
    train = np.load(train_path, allow_pickle = True)
    if model_type == 'fcn' or model_type == 'vae':
        train = train.reshape(len(train), -1)

    nepoch = 1000
    batch_size = 128
    learning_rate = 1e-3
        
    data = torch.tensor(train, dtype = torch.float)
    train_dataset = TensorDataset(data)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler = train_sampler, batch_size = batch_size)

    models = {'fcn' : fcn_AE(), 'cnn' : conv_AE(), 'vae' : VAE()}
    model = models[model_type].cuda()
    
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr = learning_rate)

    model.train()
    best_loss = np.inf
    for epoch in range(nepoch):
        for data in train_dataloader:
            if model_type == 'cnn':
                img = data[0].transpose(3, 1).cuda()
            else:
                img = data[0].cuda()
            # =================== forward =====================
            output = model(img)
            if model_type == 'vae':
                loss = loss_vae(output[0], img, mu = output[2], logvar = output[3], criterion = criterion)
            else:
                loss = criterion(output[0], img)
            # =================== backward ====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # =================== save ========================
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), model_path)
        # =================== log =============================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, nepoch, loss.item()))