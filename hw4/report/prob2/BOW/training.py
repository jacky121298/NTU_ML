import torch
from torch import nn
import numpy as np
import torch.optim as optim
from preprocess import Preprocess

def evaluation(outputs, labels):
    # outputs => probability (float)
    # labels => labels
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

def training(batch_size, n_epoch, lr, model_dir, X_train, X_val, y_train, y_val, model, device, preprocess):
    total = sum(p.numel() for p in model.parameters())
    print('\nstart training, parameter total:{}\n'.format(total))
    
    model.train() # 將 model 的模式設為 train，這樣 optimizer 就可以更新 model 的參數
    criterion = nn.BCELoss() # 定義損失函數，這裡我們使用 binary cross entropy loss
    t_batch = int(np.ceil(len(X_train) / batch_size))
    v_batch = int(np.ceil(len(X_val) / batch_size))
    optimizer = optim.Adam(model.parameters(), lr = lr) # 將模型的參數給 optimizer，並給予適當的 learning rate

    best_acc = 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        # 這段做 training
        for i in range(t_batch):
            inputs = X_train[i * batch_size : (i + 1) * batch_size]
            inputs = preprocess.BOW_vector(inputs)
            inputs = inputs.to(device)
            
            labels = y_train[i * batch_size : (i + 1) * batch_size]
            labels = preprocess.labels_to_tensor(labels)
            labels = labels.to(device, dtype = torch.float) # device為 "cuda"，將 labels 轉成 torch.cuda.FloatTensor，因為等等要餵進 criterion，所以型態要是 float
            
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
            for i in range(v_batch):
                inputs = X_val[i * batch_size : (i + 1) * batch_size]
                inputs = preprocess.BOW_vector(inputs)
                inputs = inputs.to(device)
                
                labels = y_val[i * batch_size : (i + 1) * batch_size]
                labels = preprocess.labels_to_tensor(labels)
                labels = labels.to(device, dtype = torch.float) # device為 "cuda"，將 labels 轉成 torch.cuda.FloatTensor，因為等等要餵進 criterion，所以型態要是 float
                
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
                torch.save(model, "{}/bow.model".format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
        
        print('-----------------------------------------------')
        model.train()