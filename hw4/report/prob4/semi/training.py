import torch
from torch import nn
from torch.utils import data
import torch.optim as optim

class TwitterDataset(data.Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)
    
    __len__ will return the number of data
    """
    def __init__(self, X, y):
        self.data = X
        self.label = y
    
    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]
    
    def __len__(self):
        return len(self.data)

def evaluation(outputs, labels):
    # outputs => probability (float)
    # labels => labels
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

def build_dataset(X_train, y_train, X_train_no, batch_size):
    train_dataset = TwitterDataset(X = X_train, y = y_train)
    train_no_dataset = TwitterDataset(X = X_train_no, y = None)
    
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = 8)
    train_no_loader = torch.utils.data.DataLoader(dataset = train_no_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)
    return train_loader, train_no_loader

def training(batch_size, n_epoch, lr, model_dir, X_train, X_val, y_train, y_val, X_train_no, model, device, threshold, pad_index):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    
    val_dataset = TwitterDataset(X = X_val, y = y_val)
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)
    v_batch = len(val_loader)

    model.train() # 將 model 的模式設為 train，這樣 optimizer 就可以更新 model 的參數
    criterion = nn.BCELoss() # 定義損失函數，這裡我們使用 binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr = lr) # 將模型的參數給 optimizer，並給予適當的 learning rate
    
    best_acc = 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        train_loader, train_no_loader = build_dataset(X_train, y_train, X_train_no, batch_size)
        t_batch = len(train_loader)
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, dtype = torch.long)  # device 為 "cuda"，將 inputs 轉成 torch.cuda.LongTensor
            labels = labels.to(device, dtype = torch.float) # device 為 "cuda"，將 labels 轉成 torch.cuda.FloatTensor，因為等等要餵進 criterion，所以型態要是 float
            
            optimizer.zero_grad() # 由於 loss.backward() 的 gradient 會累加，所以每次餵完一個 batch 後需要歸零
            outputs = model(inputs) # 將 input 餵給模型
            outputs = outputs.squeeze() # 去掉最外面的 dimension，好讓 outputs 可以餵進 criterion()
            loss = criterion(outputs, labels) # 計算此時模型的 training loss
            loss.backward() # 算 loss 的 gradient
            optimizer.step() # 更新訓練模型的參數
            
            correct = evaluation(outputs, labels) # 計算此時模型的 training accuracy
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(epoch+1, i+1, t_batch, loss.item(), correct*100/batch_size), end = '\r')
        
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch*100))

        # 這段做 validation
        model.eval() # 將 model 的模式設為 eval，這樣 model 的參數就會固定住
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device, dtype = torch.long) # device 為 "cuda"，將 inputs 轉成 torch.cuda.LongTensor
                labels = labels.to(device, dtype = torch.float) # device 為 "cuda"，將 labels 轉成 torch.cuda.FloatTensor，因為等等要餵進 criterion，所以型態要是 float
                outputs = model(inputs) # 將 input 餵給模型
                outputs = outputs.squeeze() # 去掉最外面的 dimension，好讓 outputs 可以餵進 criterion()
                loss = criterion(outputs, labels) # 計算此時模型的 validation loss
                correct = evaluation(outputs, labels) # 計算此時模型的 validation accuracy
                
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            
            if total_acc > best_acc:
                # 如果 validation 的結果優於之前所有的結果，就把當下的模型存下來以備之後做預測時使用
                best_acc = total_acc
                #torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
                torch.save(model, "{}/ckpt.model".format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))

            print("Semi-supervised learning ...")
            for i, inputs in enumerate(train_no_loader):
                inputs = inputs.to(device, dtype = torch.long)
                outputs = model(inputs)
                outputs = outputs.squeeze()
                
                for j in range(len(outputs)):
                    if X_train_no[i * batch_size + j][0] == pad_index:
                        continue

                    if outputs[j].item() >= threshold:
                        X_train = torch.cat((X_train, X_train_no[[i * batch_size + j], :]), 0)
                        y_train = torch.cat((y_train, torch.ones(1, dtype = torch.long)), 0)
                        X_train_no[i * batch_size + j][0] = pad_index

                    elif outputs[j].item() <= (1 - threshold):
                        X_train = torch.cat((X_train, X_train_no[[i * batch_size + j], :]), 0)
                        y_train = torch.cat((y_train, torch.zeros(1, dtype = torch.long)), 0)
                        X_train_no[i * batch_size + j][0] = pad_index

                    print("Now it's processing the {}-th unlabel data".format(i * batch_size + j), end = '\r')
    
            print("The shape of X_train becomes to {}\n".format(X_train.shape))
        
        print('-----------------------------------------------')
        model.train()