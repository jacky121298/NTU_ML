import sys
import torch
import random
import numpy as np
import torch.nn as nn
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
    transformer = KernelPCA(n_components = 200, kernel = 'rbf', random_state = 0, n_jobs = -1)
    kpca = transformer.fit_transform(latents)
    print('Reduction Shape:', kpca.shape)

    # pca = PCA(n_components = 64, random_state = 0).fit_transform(kpca)
    # print('Reduction Shape:', pca.shape)
    # pca = PCA(n_components = 16, random_state = 0).fit_transform(pca)
    # print('Reduction Shape:', pca.shape)

    # Dimesnion Reduction
    X_embedded = TSNE(n_components = 2).fit_transform(kpca)
    print('Reduction Shape:', X_embedded.shape)

    # Clustering
    pred = MiniBatchKMeans(n_clusters = 2, random_state = 0).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred, X_embedded

def invert(pred):
    return np.abs(1 - pred)

def save_prediction(pred, out_csv = 'prediction.csv'):
    with open(out_csv, 'w') as f:
        f.write('id,label\n')
        for i, p in enumerate(pred):
            f.write(f'{i},{p}\n')
    print(f'Save prediction to {out_csv}')

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
    out_csv = sys.argv[3]

    same_seeds(seed = 0)
    model = AE().cuda()
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    trainX = np.load(trainX_path)
    latents = inference(X = trainX, model = model)
    pred, X_embedded = predict(latents)

    save_prediction(pred, out_csv = out_csv)
    # save_prediction(invert(pred), out_csv = 'submit.csv')