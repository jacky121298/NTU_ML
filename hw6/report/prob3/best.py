import os
import sys
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        self.root = root
        self.label = torch.from_numpy(label).long()
        self.transforms = transforms
        self.fnames = []

        for i in range(200):
            self.fnames.append('{:03d}'.format(i))

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.fnames[idx] + '.png'))
        img = self.transforms(img)
        label = self.label[idx]
        return img, label
    
    def __len__(self):
        return 200

class Attacker:
    def __init__(self, img_dir, label):
        self.model = models.densenet121(pretrained = True)
        self.model.cuda()
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        
        transform = transforms.Compose([                
            transforms.Resize((224, 224), interpolation = 3),
            transforms.ToTensor(),
            self.normalize
        ])    
        self.dataset = Adverdataset(img_dir, label, transform)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size = 1, shuffle = False)

    def fgsm_attack(self, image, epsilon, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        return perturbed_image

    def attack(self, plt_img, label_name):
        adversarial = []
        device = torch.device('cuda')

        i = 0
        for (data, target) in self.loader:
            data, target = data.to(device), target.to(device)
            data.requires_grad = True
            output = self.model(data)
            init_pred = output.max(1, keepdim = True)[1]
            
            softmax = torch.nn.Softmax(dim = 1)
            output = softmax(output)
            prob_1, class_1 = torch.sort(output, dim = 1, descending = True)
            prob_1 = prob_1.squeeze()[:3].detach().cpu().numpy()
            class_1 = class_1.squeeze()[:3].detach().cpu().numpy()
            
            success = False
            for eps in range(1, 150):
                loss = F.nll_loss(output, target)
                self.model.zero_grad()
                loss.backward()
                data_grad = data.grad.data
                perturbed_data = self.fgsm_attack(data, eps / 100, data_grad)

                output = self.model(perturbed_data)
                final_pred = output.max(1, keepdim = True)[1]
                output = softmax(output)
                
                prob_2, class_2 = torch.sort(output, dim = 1, descending = True)
                prob_2 = prob_2.squeeze()[:3].detach().cpu().numpy()
                class_2 = class_2.squeeze()[:3].detach().cpu().numpy()

                if final_pred.item() != target.item():
                    success = True
                    break

            if success: adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            else: adv_ex = data.squeeze().detach().cpu().numpy()
            adv_ex = np.clip((np.transpose(adv_ex, (1, 2, 0)) * self.std + self.mean) * 255, 0, 255).astype('uint8')
            adversarial.append(adv_ex)

            if i in plt_img:
                fig, axs = plt.subplots(1, 3, figsize = (15, 8))
                
                axs[0].axis('off')
                axs[0].imshow(adv_ex)
                axs[0].set_title('img_{:03d}'.format(i))

                x = [0, 1, 2]
                axs[1].bar(x, prob_1, width = 0.5, tick_label = [label_name[c].split(',')[0] for c in class_1])
                axs[1].set_title('Original Image')

                axs[2].bar(x, prob_2, width = 0.5, tick_label = [label_name[c].split(',')[0] for c in class_2])
                axs[2].set_title('Adversarial Image')

                plt.savefig('img_{:03d}.png'.format(i))
                plt.close()
            
            i += 1
            print('it\'s processing the {}-th image'.format(i), end = '\n' if i == len(self.loader) else '\r')

        return adversarial

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_img = sys.argv[2]

    label = pd.read_csv('{}/labels.csv'.format(input_dir))
    label = label.loc[:, 'TrueLabel'].to_numpy()
    label_name = pd.read_csv('{}/categories.csv'.format(input_dir))
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()
    
    plt_img = [98, 121, 192]
    attacker = Attacker('{}/images'.format(input_dir), label)
    adversarial = attacker.attack(plt_img, label_name)

    for i in range(len(adversarial)):
        img = Image.fromarray(adversarial[i])
        img.save('{}/{:03d}.png'.format(output_img, i))