import os
import sys
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from glob import glob
from PIL import Image
from model import StudentNet

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, folderName, transform = None):
        self.transform = transform
        self.data, self.label = [], []

        for img_path in sorted(glob(folderName + '/*.jpg')):
            try:
                class_idx = int(re.findall(re.compile(r'\d+'), img_path)[1])
            except:
                class_idx = 0

            image = Image.open(img_path)
            image_fp = image.fp
            image.load()
            image_fp.close()

            self.data.append(image)
            self.label.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.label[idx]

def get_dataloader(mode = 'training', batch_size = 32, data_dir = './food-11'):
    assert mode in ['training', 'testing', 'validation']
    dataset = MyDataset(os.path.join(data_dir, mode), transform = trainTransform if mode == 'training' else testTransform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = (mode == 'training'))
    return dataloader

def network_slimming(old_model, new_model):
    params = old_model.state_dict()
    new_params = new_model.state_dict()
    
    selected_idx = []
    for i in range(8):
        importance = params[f'cnn.{i}.1.weight']
        old_dim = len(importance)
        new_dim = len(new_params[f'cnn.{i}.1.weight'])
        ranking = torch.argsort(importance, descending = True)
        selected_idx.append(ranking[:new_dim])

    now_processed = 3
    for (name, p1), (name2, p2) in zip(params.items(), new_params.items()):
        if name.startswith(f'cnn.{now_processed}') and p1.size() != torch.Size([]) and now_processed != len(selected_idx):
            if name.endswith('0.weight'):
                if now_processed == len(selected_idx) - 1:
                    new_params[name] = p1[:,selected_idx[now_processed-1]]
                else:
                    new_params[name] = p1[selected_idx[now_processed]][:,selected_idx[now_processed-1]]
            else:
                new_params[name] = p1[selected_idx[now_processed]]
        else:
            new_params[name] = p1

        if name.endswith(f'{now_processed}.1.num_batches_tracked'):
            now_processed += 1

    new_model.load_state_dict(new_params)
    return new_model

def run_epoch(dataloader, update = True, alpha = 0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        optimizer.zero_grad()
        inputs, labels = batch_data
        inputs, labels = inputs.cuda(), labels.cuda()
  
        logits = net(inputs)
        loss = criterion(logits, labels)
        if update:
            loss.backward()
            optimizer.step()

        total_hit += torch.sum(torch.argmax(logits, dim = 1) == labels).item()
        total_num += len(inputs)
        total_loss += loss.item() * len(inputs)

    return (total_loss / total_num), (total_hit / total_num)

if __name__ == '__main__':
    data_dir = sys.argv[1]
    student_model = sys.argv[2]
    pruned_dir = sys.argv[3]

    trainTransform = transforms.Compose([
        transforms.RandomCrop(256, pad_if_needed = True, padding_mode = 'symmetric'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    testTransform = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    batch_size = 32
    print('reading training data ...')
    train_dataloader = get_dataloader('training', batch_size = batch_size, data_dir = data_dir)
    print('reading validation data ...')
    valid_dataloader = get_dataloader('validation', batch_size = batch_size, data_dir = data_dir)

    net = StudentNet().cuda()
    net.load_state_dict(torch.load(student_model))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr = 1e-3)

    nepoch = 5
    width_mult = [0.23]
    
    print('start network pruning ...')
    for now_width_mult in width_mult:
        new_net = StudentNet(width_mult = now_width_mult).cuda()
        net = network_slimming(net, new_net)
        now_best_acc = 0
        
        for epoch in range(nepoch):
            net.train()
            train_loss, train_acc = run_epoch(train_dataloader, update = True)
            net.eval()
            valid_loss, valid_acc = run_epoch(valid_dataloader, update = False)
            
            if valid_acc > now_best_acc:
                now_best_acc = valid_acc
                torch.save(net.state_dict(), f'{pruned_dir}/pruned_{now_width_mult}.bin')
            
            print('rate {:6.4f} || epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} | valid loss: {:6.4f}, acc {:6.4f}'.format(
                now_width_mult, epoch, train_loss, train_acc, valid_loss, valid_acc))