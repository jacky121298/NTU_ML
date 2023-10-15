import sys
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from autoencoder import AE

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

if __name__ == '__main__':
    trainX_path = sys.argv[1]
    checkpoint = sys.argv[2]

    trainX = np.load(trainX_path)
    trainX_p = preprocess(trainX)
    plt.figure(figsize = (10, 4))
    
    indexes = [1, 2, 3, 6, 7, 9]
    imgs = trainX[indexes,]

    model = AE().cuda()
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    
    for i, img in enumerate(imgs):
        plt.subplot(2, len(indexes), i + 1, xticks = [], yticks = [])
        plt.imshow(img)

    inp = torch.Tensor(trainX_p[indexes,]).cuda()
    latents, recs = model(inp)
    recs = ((recs + 1) / 2).cpu().detach().numpy()
    recs = recs.transpose(0, 2, 3, 1)
    
    for i, img in enumerate(recs):
        plt.subplot(2, len(indexes), len(indexes) + i + 1, xticks = [], yticks = [])
        plt.imshow(img)
    
    plt.tight_layout()
    plt.savefig('reconstruct.png')