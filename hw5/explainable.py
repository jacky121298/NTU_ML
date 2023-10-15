import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.segmentation import slic
from lime import lime_image

def readfile(path):
    imgnames = sorted(os.listdir(path))
    imgpaths, labels = [], []
    for name in imgnames:
        imgpaths.append(os.path.join(path, name))
        labels.append(int(name.split('_')[0]))
    return imgpaths, labels

class ImgDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        X = Image.open(self.paths[index])
        X = self.transform(X)
        Y = self.labels[index]
        return X, Y

    # 這個 method 並不是 pytorch dataset 必要，只是方便未來我們想要指定「取哪幾張圖片」出來當作一個 batch 來 visualize
    def getbatch(self, indices):
        images, labels = [], []
        for index in indices:
            image, label = self.__getitem__(index)
            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        def building_block(indim, outdim):
            return [
                nn.Conv2d(indim, outdim, 3, 1, 1),
                nn.BatchNorm2d(outdim),
                nn.PReLU(),
            ]

        def stack_blocks(indim, outdim, block_num):
            layers = building_block(indim, outdim)
            for i in range(block_num - 1):
                layers += building_block(outdim, outdim)
            layers.append(nn.MaxPool2d(2, 2, 0))
            return layers

        cnn_list = []
        cnn_list += stack_blocks(3, 64, 2)
        cnn_list += stack_blocks(64, 128, 1)
        cnn_list += stack_blocks(128, 256, 1)
        cnn_list += stack_blocks(256, 512, 1)
        cnn_list += stack_blocks(512, 512, 1)
        self.cnn = nn.Sequential(*cnn_list)

        dnn_list = [
            nn.Linear(512 * 4 * 4, 1024),
            nn.Dropout(p = 0.5, inplace = True),
            nn.PReLU(),

            nn.Linear(1024, 512),
            nn.Dropout(p = 0.5, inplace = True),
            nn.PReLU(),
            nn.Linear(512, 11)
        ]
        self.fc = nn.Sequential(*dnn_list)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())

def compute_saliency_maps(x, y, model):
    model.eval()
    x = x.cuda()
    x.requires_grad_()
    
    y_pred = model(x)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()
    
    saliencies = x.grad.abs().detach().cpu()
    saliencies = torch.stack([normalize(item) for item in saliencies])
    return saliencies

layer_activations = None
def filter_explaination(x, model, cnnid, filterid, iteration, lr):
    model.eval()

    def hook(model, input, output):
        global layer_activations
        layer_activations = output
    hook_handle = model.cnn[cnnid].register_forward_hook(hook)

    # Filter activation: 我們先觀察 x 經過被指定 filter 的 activation map
    model(x.cuda())
    filter_activations = layer_activations[:, filterid, :, :].detach().cpu()
  
    # Filter visualization: 接著我們要找出可以最大程度 activate 該 filter 的圖片
    x = x.cuda()
    x.requires_grad_()
    optimizer = Adam([x], lr = lr)
    
    for iter in range(iteration):
        optimizer.zero_grad()
        model(x)
        objective = -layer_activations[:, filterid, :, :].sum()
        objective.backward()
        optimizer.step()
    
    filter_visualization = x.detach().cpu().squeeze()[0]
    hook_handle.remove()
    return filter_activations, filter_visualization

def predict(input):
    # input: numpy array, (batches, height, width, channels)
    model.eval()
    input = torch.FloatTensor(input).permute(0, 3, 1, 2)
    output = model(input.cuda())
    return output.detach().cpu().numpy()

def segmentation(input):
    # 利用 skimage 提供的 segmentation 將圖片分成 100 塊
    return slic(input, n_segments = 100, compactness = 1, sigma = 1)

if __name__ == '__main__':
    dataset = sys.argv[1]
    model_path = sys.argv[2]
    output_img = sys.argv[3]

    print("Loading model ...")
    model = Classifier().cuda()
    model.load_state_dict(torch.load(model_path))

    print("Reading data ...")
    train_paths, train_labels = readfile(os.path.join(dataset, "training"))
    print("Size of training data = {}".format(len(train_paths)))

    # fix the random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    
    # testing : no data augmentation
    test_transform = transforms.Compose([
        transforms.Resize(size = (128, 128)),
        transforms.ToTensor(),
    ])
    train_set = ImgDataset(train_paths, train_labels, test_transform)

    ########################################################################################
    # Saliency map
    # saliencies: (batches, channels, height, weight)
    # 因為接下來我們要對每張圖片畫 saliency map，每張圖片的 gradient scale 很可能有巨大落差
    # 可能第一張圖片的 gradient 在 100 ~ 1000，但第二張圖片的 gradient 在 0.001 ~ 0.0001
    # 如果我們用同樣的色階去畫每一張 saliency 的話，第一張可能就全部都很亮，第二張就全部都很暗
    # 如此就看不到有意義的結果，我們想看的是「單一張 saliency 內部的大小關係」，
    # 所以這邊我們要對每張 saliency 各自做 normalize。手法有很多種，這邊只採用最簡單的
    ########################################################################################
    indices = [1, 2309, 8197, 9040]
    images, labels = train_set.getbatch(indices)
    saliencies = compute_saliency_maps(images, labels, model)

    # 使用 matplotlib 畫出來
    # 在 pytorch 的世界，image tensor 各 dimension 的意義通常為 (channels, height, width)
    # 在 matplotlib 的世界，想要把一個 tensor 畫出來，形狀必須為 (height, width, channels)
    fig, axs = plt.subplots(2, len(indices), figsize = (15, 8))
    for row, target in enumerate([images, saliencies]):
        for column, img in enumerate(target):
            axs[row][column].imshow(img.permute(1, 2, 0).numpy())

    plt.savefig('{}/saliency.png'.format(output_img))
    plt.close()

    ########################################################################################
    # Filter explaination
    # hook_handle = model.cnn[cnnid].register_forward_hook(hook)
    # 這一行是在告訴 pytorch，當 forward 「過了」第 cnnid 層 cnn 後，要先呼叫 hook 這個我們定義
    # 的 function 後才可以繼續 forward 下一層 cnn 因此上面的 hook function 中，我們就會把該層的
    # output，也就是 activation map 記錄下來，這樣 forward 完整個 model 後我們就不只有 loss
    # 也有某層 cnn 的 activation map
    ########################################################################################
    cnnid = 9
    filterid = 10

    # filter visualization
    filter_activations, filter_visualization = filter_explaination(images, model, cnnid, filterid, iteration = 100, lr = 0.1)
    plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
    plt.savefig('{}/visualization.png'.format(output_img))
    plt.close()

    # filter activations
    fig, axs = plt.subplots(2, len(indices), figsize = (15, 8))
    for i, img in enumerate(images):
        axs[0][i].imshow(img.permute(1, 2, 0))
    for i, img in enumerate(filter_activations):
        axs[1][i].imshow(normalize(img))
    
    plt.savefig('{}/activations.png'.format(output_img))
    plt.close()

    ########################################################################################
    # Lime
    # 基本上只要提供給 lime explainer 兩個關鍵的 function，事情就結束了
    # classifier_fn 定義圖片如何經過 model 得到 prediction
    # segmentation_fn 定義如何把圖片做 segmentation
    ########################################################################################
    fig, axs = plt.subplots(1, len(indices), figsize = (15, 8))
    for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
        x = image.astype(np.double)
        # lime 這個套件要吃 numpy array
        explainer = lime_image.LimeImageExplainer()
        explaination = explainer.explain_instance(image = x, top_labels = 11, classifier_fn = predict, segmentation_fn = segmentation)

        lime_img, mask = explaination.get_image_and_mask(
            label = label.item(),
            positive_only = False,
            hide_rest = False,
            num_features = 11,
            min_weight = 0.05
        )
        axs[idx].imshow(lime_img)
    
    plt.savefig('{}/lime.png'.format(output_img))
    plt.close()