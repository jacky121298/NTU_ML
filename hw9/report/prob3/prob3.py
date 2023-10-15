import sys
import glob
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
    trainX_path = sys.argv[3]
    checkpoints = sys.argv[4]

    valX = np.load(valX_path)
    valY = np.load(valY_path)
    trainX = np.load(trainX_path)
    
    trainX = preprocess(trainX)
    checkpoints_list = sorted(glob.glob(f'{checkpoints}/checkpoint_*.pth'))

    dataset = Image_Dataset(trainX)
    dataloader = DataLoader(dataset, batch_size = 64, shuffle = False)

    same_seeds(seed = 0)
    points = []
    model = AE().cuda()
    
    with torch.no_grad():
        for i, checkpoint in enumerate(checkpoints_list):
            print('[{}/{}] {}'.format(i + 1, len(checkpoints_list), checkpoint))
            model.load_state_dict(torch.load(checkpoint))
            model.eval()
            
            err, n = 0, 0
            for x in dataloader:
                x = x.cuda()
                _, rec = model(x)
                err += torch.nn.MSELoss(reduction = 'sum')(x, rec).item()
                n += x.flatten().size(0)
            
            print('Reconstruction error (MSE):', err / n)
            latents = inference(X = valX, model = model)
            pred, X_embedded = predict(latents)
            
            acc = cal_acc(valY, pred)
            print('Accuracy:', acc)
            points.append((err / n, acc))

    ps = list(zip(* points))
    plt.figure(figsize = (6, 6))
    xticks = [i for i in range(20, 220, 20)]
    
    plt.subplot(211, title = 'Reconstruction error (MSE)').plot(ps[0])
    plt.xlabel('# of epoch')
    plt.xticks(ticks = np.arange(10), labels = xticks)

    plt.subplot(212, title = 'Accuracy (val)').plot(ps[1])
    plt.xlabel('# of epoch')
    plt.xticks(ticks = np.arange(10), labels = xticks)
    
    plt.tight_layout()
    plt.savefig('checkpoints.png')