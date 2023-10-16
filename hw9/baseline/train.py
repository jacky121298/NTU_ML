import sys
import torch
import random
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from autoencoder import AE

class Image_Dataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        images = self.image_list[idx]
        return images

def preprocess(image_list):
    """ Normalize Image and Permute (N, H, W, C) to (N, C, H, W)
    Args:
      image_list: List of images (9000, 32, 32, 3)
    Returns:
      image_list: List of images (9000, 3, 32, 32)
    """
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    image_list = (image_list / 255.0) * 2 - 1
    image_list = image_list.astype(np.float32)
    return image_list

def count_parameters(model, only_trainable = False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def same_seeds(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
    
    torch.manual_seed(seed)
    np.random.seed(seed) # numpy module
    random.seed(seed) # python random module
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    trainX_path = sys.argv[1]
    checkpoint = sys.argv[2]

    trainX = np.load(trainX_path)
    trainX = preprocess(trainX)
    dataset = Image_Dataset(trainX)

    seed = 0
    same_seeds(seed)

    model = AE().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5, weight_decay = 1e-5)

    n_epoch = 100
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    model.train()
    for epoch in range(n_epoch):
        for data in dataloader:
            img = data.cuda()
            output1, output = model(img)
            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if (epoch + 1) % 20 == 0:
            #    torch.save(model.state_dict(), '{}/checkpoint_{:03d}.pth'.format(checkpoints, epoch + 1))
                
        print('epoch [{}/{}], loss:{:.5f}'.format(epoch + 1, n_epoch, loss.data))

    torch.save(model.state_dict(), checkpoint)