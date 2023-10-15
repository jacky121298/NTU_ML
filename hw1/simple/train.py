import sys
import pandas as pd
import numpy as np

def readData(path):
    data = pd.read_csv(path, encoding = 'big5').iloc[:, 3:].replace('NR', 0)
    raw_data = data.to_numpy(dtype = float)

    month_data = {}
    for month in range(12):
        tmp = np.empty([18, 480])
        for day in range(20):
            tmp[:, 24 * day : 24 * (day + 1)] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
        month_data[month] = tmp

    train_x = np.empty([471 * 12, 18 * 9], dtype = float)
    train_y = np.empty([471 * 12, 1], dtype = float)

    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    continue
                train_x[471 * month + 24 * day + hour, :] = month_data[month][:, 24 * day + hour : 24 * day + hour + 9].reshape(1, -1)
                train_y[471 * month + 24 * day + hour, 0] = month_data[month][9, 24 * day + hour + 9]
    
    mean_x = np.mean(train_x, axis = 0) # 18 * 9
    std_x = np.std(train_x, axis = 0) # 18 * 9
    for i in range(len(train_x)):
        for j in range(len(train_x[0])):
            if std_x[j] != 0:
                train_x[i][j] = (train_x[i][j] - mean_x[j]) / std_x[j]

    return train_x, train_y, mean_x, std_x

def rmse(x, y, w):
    return np.sqrt( np.sum( ( np.dot(x, w) - y) ** 2 ) / 471 / 12 )

def gradientDescent(train_x, train_y):
    dim = 18 * 9 + 1
    w = np.zeros([dim, 1])
    train_x = np.concatenate( (np.ones([471 * 12, 1]), train_x), axis = 1 ).astype(float)
    learning_rate = 1
    eps = 0.0000000001
    iter_time = 2000
    adagrad = np.zeros([dim, 1])
    for t in range(iter_time):
        loss = rmse(train_x, train_y, w)
        # if t % 100 == 0:
        #    print( str(t), ':', str(loss) )
        gradient = 2 * np.dot( train_x.transpose(), np.dot(train_x, w) - train_y)
        adagrad += ( gradient ** 2 )
        w -= ( learning_rate * gradient / np.sqrt(adagrad + eps) )
    
    return w

if __name__ == '__main__':
    train_csv = sys.argv[1]
    np_file = sys.argv[2]
    
    train_x, train_y, mean_x, std_x = readData(train_csv)
    w = gradientDescent(train_x, train_y)

    np.savez(np_file, w = w, mean_x = mean_x, std_x = std_x)