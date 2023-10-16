import os
import sys
import re
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from glob import glob
from PIL import Image
from architecuture import StudentNet

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform = None):
        self.transform = transform
        self.data, self.label = X, y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)

        if self.label is None:
            return image
        else: 
            return image, self.label[idx]

def loss_fn_kd(outputs, labels, teacher_outputs, T = 20, alpha = 0.5):
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    soft_loss = nn.KLDivLoss(reduction = 'batchmean')(F.log_softmax(outputs/T, dim = 1), F.softmax(teacher_outputs/T, dim = 1)) * (alpha * T * T)
    return hard_loss + soft_loss

def read_data(mode = 'training', data_dir = './food-11', labeled = True):
    data, label = [], []
    for img_path in sorted(glob(os.path.join(data_dir, mode) + '/*.jpg')):
        if labeled:
            class_idx = int(re.findall(re.compile(r'\d+'), img_path)[1])
            label.append(class_idx)

        image = Image.open(img_path)
        image_fp = image.fp
        image.load()
        image_fp.close()
        data.append(image)
    
    return data, label

def get_dataloader(X, y, batch_size = 32, mode = 'training'):
    dataset = MyDataset(X, y, transform = trainTransform if mode == 'training' else testTransform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = (mode == 'training'))
    return dataloader

def run_student(dataloader, update = True, alpha = 0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        opt_student.zero_grad()
        inputs, hard_labels = batch_data
        
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        with torch.no_grad():
            soft_labels = teacher_net(inputs)

        if update:
            logits = student_net(inputs)
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            loss.backward()
            opt_student.step()    
        else:
            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            
        total_hit += torch.sum(torch.argmax(logits, dim = 1) == hard_labels).item()
        total_num += len(inputs)
        total_loss += loss.item() * len(inputs)
    
    return (total_loss / total_num), (total_hit / total_num)

def run_teacher(dataloader, update = True, alpha = 0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        opt_teacher.zero_grad()
        inputs, hard_labels = batch_data
        
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        with torch.no_grad():
            soft_labels = student_net(inputs)

        if update:
            logits = teacher_net(inputs)
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            loss.backward()
            opt_teacher.step()
        else:
            with torch.no_grad():
                logits = teacher_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            
        total_hit += torch.sum(torch.argmax(logits, dim = 1) == hard_labels).item()
        total_num += len(inputs)
        total_loss += loss.item() * len(inputs)
    
    return (total_loss / total_num), (total_hit / total_num)

if __name__ == '__main__':
    data_dir = sys.argv[1]
    teacher_model = sys.argv[2]
    student_dir = sys.argv[3]

    # fix the random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

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

    print('reading training data ...')
    train_x, train_y = read_data('training', data_dir, labeled = True)
    print('reading validation data ...')
    val_x, val_y = read_data('validation', data_dir, labeled = True)
    print('reading testing data ...')
    test_x, _ = read_data('testing', data_dir, labeled = False)

    batch_size = 32
    valid_dataloader = get_dataloader(val_x, val_y, batch_size = batch_size, mode = 'validation')
    test_dataloader = get_dataloader(test_x, None, batch_size = batch_size, mode = 'testing')

    teacher_net = models.resnet18(pretrained = False, num_classes = 11).cuda()
    student_net = StudentNet(base = 16).cuda()

    teacher_net.load_state_dict(torch.load(teacher_model))
    opt_student = optim.AdamW(student_net.parameters(), lr = 1e-3)
    opt_teacher = optim.AdamW(teacher_net.parameters(), lr = 1e-5)

    nstudent = 300
    nteacher = 50
    now_best_acc = 0
    dic_test = {}
    teacher_net.eval()
    
    print('start knowledge distillation ...')
    for estudent in range(nstudent):
        train_dataloader = get_dataloader(train_x, train_y, batch_size = batch_size, mode = 'training')
        student_net.train()
        train_loss, train_acc = run_student(train_dataloader, update = True, alpha = 0.5)
        student_net.eval()
        valid_loss, valid_acc = run_student(valid_dataloader, update = False, alpha = 0.5)

        if valid_acc > now_best_acc:
            now_best_acc = valid_acc
            torch.save(student_net.state_dict(), f'{student_dir}/student_net.bin')
            
        print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} | valid loss: {:6.4f}, acc {:6.4f}'.format(estudent + 1, train_loss, train_acc, valid_loss, valid_acc))
        for now_step, inputs in enumerate(test_dataloader):        
            inputs = inputs.cuda()
            logits = student_net(inputs)
            outputs = F.softmax(logits, dim = 1)
            outputs = outputs.cpu().detach().numpy()
            output = np.argmax(outputs, axis = 1)

            labels = teacher_net(inputs)
            labels = F.softmax(labels, dim = 1)
            labels = labels.cpu().detach().numpy()
            label = np.argmax(labels, axis = 1)
            
            for idx in range(len(output)):
                if (now_step * batch_size + idx) not in dic_test.keys() and outputs[idx][output[idx]] >= 0.95 and (label[idx] == output[idx]):
                    train_x.append(test_x[now_step * batch_size + idx])
                    train_y.append(output[idx])
                    dic_test[now_step * batch_size + idx] = 1
        print("semi-supervised learning, size becomes to", len(train_x))

    for eteacher in range(nteacher):
        teacher_net.train()
        train_loss, train_acc = run_teacher(train_dataloader, update = True, alpha = 0.5)
        teacher_net.eval()
        valid_loss, valid_acc = run_teacher(valid_dataloader, update = False, alpha = 0.5)
            
        print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} | valid loss: {:6.4f}, acc {:6.4f}'.format(eteacher + 1, train_loss, train_acc, valid_loss, valid_acc))

######################################################################################################################################################################
## use another optimizer to train again                                                                                                                              ##
######################################################################################################################################################################
    opt_student = optim.SGD(student_net.parameters(), lr = 1e-3)
    opt_teacher = optim.SGD(teacher_net.parameters(), lr = 1e-5)

    print('\n====== second round ======\n')
    for estudent in range(nstudent):
        student_net.train()
        train_loss, train_acc = run_student(train_dataloader, update = True, alpha = 0.5)
        student_net.eval()
        valid_loss, valid_acc = run_student(valid_dataloader, update = False, alpha = 0.5)

        if valid_acc > now_best_acc:
            now_best_acc = valid_acc
            torch.save(student_net.state_dict(), f'{student_dir}/student_net.bin')
            
        print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} | valid loss: {:6.4f}, acc {:6.4f}'.format(estudent + 1, train_loss, train_acc, valid_loss, valid_acc))
        for now_step, inputs in enumerate(test_dataloader):        
            inputs = inputs.cuda()
            logits = student_net(inputs)
            output = F.softmax(logits, dim = 1)
            output = output.cpu().detach().numpy()
            
            for idx in range(len(output)):
                if (now_step * batch_size + idx) not in dic_test.keys() and output[idx].max() >= 0.95:
                    train_x.append(test_x[now_step * batch_size + idx])
                    train_y.append(np.argmax(output[idx]))
                    dic_test[now_step * batch_size + idx] = 1
        print("semi-supervised learning, size becomes to", len(train_x))

    for eteacher in range(nteacher):
        teacher_net.train()
        train_loss, train_acc = run_teacher(train_dataloader, update = True, alpha = 0.5)
        teacher_net.eval()
        valid_loss, valid_acc = run_teacher(valid_dataloader, update = False, alpha = 0.5)
            
        print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} | valid loss: {:6.4f}, acc {:6.4f}'.format(eteacher + 1, train_loss, train_acc, valid_loss, valid_acc))