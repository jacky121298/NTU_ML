import sys
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import KernelPCA, PCA
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

def inference(X, model, batch_size = 256):
    X = preprocess(X)
    dataset = Image_Dataset(X)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
    
    latents = []
    for i, x in enumerate(dataloader):
        x = torch.FloatTensor(x)
        vec, img = model(x.cuda())
        if i == 0:
            latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
        else:
            latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis = 0)
    
    print('Latents Shape:', latents.shape)
    return latents

def predict(latents):
    # Dimension Reduction
    transformer = KernelPCA(n_components = 500, kernel = 'rbf', random_state = 0, n_jobs = -1)
    kpca = transformer.fit_transform(latents)
    print('Reduction Shape:', kpca.shape)

    pca = PCA(n_components = 64, random_state = 0).fit_transform(kpca)
    print('Reduction Shape:', pca.shape)
    pca = PCA(n_components = 16, random_state = 0).fit_transform(pca)
    print('Reduction Shape:', pca.shape)

    # Dimesnion Reduction
    X_embedded = TSNE(n_components = 2).fit_transform(pca)
    print('Reduction Shape:', X_embedded.shape)

    # Clustering
    pred = MiniBatchKMeans(n_clusters = 2, random_state = 0).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred, X_embedded

def plot_scatter(feat, label):
    """ Plot Scatter Image.
    Args:
      feat: the (x, y) coordinate of clustering result, shape: (9000, 2)
      label: ground truth label of image (0/1), shape: (9000,)
    """
    X = feat[:, 0]
    Y = feat[:, 1]
    
    plt.figure(figsize = (4, 4))
    plt.scatter(X, Y, c = label)
    plt.title('after improvement')
    plt.legend(['val_X'], loc = 'best')
    
    plt.tight_layout()
    plt.savefig('embedding.png')

def cal_acc(gt, pred):
    """ Computes categorization accuracy of our task.
    Args:
      gt: Ground truth labels (9000, )
      pred: Predicted labels (9000, )
    Returns:
      acc: Accuracy (0~1 scalar)
    """
    # Calculate Correct predictions
    correct = np.sum(gt == pred)
    acc = correct / gt.shape[0]
    return max(acc, 1 - acc)

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
    valX_path = sys.argv[1]
    valY_path = sys.argv[2]
    model_path = sys.argv[3]

    valX = np.load(valX_path)
    valY = np.load(valY_path)

    same_seeds(seed = 0)
    model = AE().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    latents = inference(valX, model)
    pred, emb = predict(latents)
    acc = cal_acc(valY, pred)
    print('The clustering accuracy is:', acc)
    plot_scatter(emb, valY)