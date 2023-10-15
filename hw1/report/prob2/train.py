import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    x_train = train_x[:math.floor(len(train_x) * 0.8), :]
    x_valid = train_x[math.floor(len(train_x) * 0.8):, :]
    y_train = train_y[:math.floor(len(train_y) * 0.8), :]
    y_valid = train_y[math.floor(len(train_y) * 0.8):, :]

    return x_train, x_valid, y_train, y_valid, mean_x, std_x

def rmse(x, y, w):
    return np.sqrt( np.sum( ( np.dot(x, w) - y) ** 2 ) / len(x) )

def gradientDescent(x_train, x_valid, y_train, y_valid):
    w_5 = np.zeros([18 * 5 + 1, 1])
    w_9 = np.zeros([18 * 9 + 1, 1])
    learning_rate = 50
    eps = 0.0000000001
    iter_time = 10000
    adagrad_5 = np.zeros([18 * 5 + 1, 1])
    adagrad_9 = np.zeros([18 * 9 + 1, 1])

    to_d = []
    for i in range(len(x_train[0])):
        if i % 9 in [0, 1, 2, 3]:
            to_d.append(i)

    x_train_9 = np.concatenate( (np.ones([len(x_train), 1]), x_train), axis = 1 ).astype(float)
    x_train_5 = np.concatenate( (np.ones([len(x_train), 1]), np.delete(x_train, to_d, axis = 1)), axis = 1 ).astype(float)
    x_valid_9 = np.concatenate( (np.ones([len(x_valid), 1]), x_valid), axis = 1 ).astype(float)
    x_valid_5 = np.concatenate( (np.ones([len(x_valid), 1]), np.delete(x_valid, to_d, axis = 1)), axis = 1 ).astype(float)

    point_x = []
    point_y = [[] for _ in range(2)]

    for t in range(iter_time):
        # if t % 100 == 0:
        #    print( str(t), ':', str(loss) )
        gradient_5 = 2 * np.dot( x_train_5.transpose(), np.dot(x_train_5, w_5) - y_train)
        adagrad_5 += ( gradient_5 ** 2 )
        w_5 -= ( learning_rate * gradient_5 / np.sqrt(adagrad_5 + eps) )

        gradient_9 = 2 * np.dot( x_train_9.transpose(), np.dot(x_train_9, w_9) - y_train)
        adagrad_9 += ( gradient_9 ** 2 )
        w_9 -= ( learning_rate * gradient_9 / np.sqrt(adagrad_9 + eps) )

        if t % 10 == 0:
            loss_5 = rmse(x_valid_5, y_valid, w_5)
            loss_9 = rmse(x_valid_9, y_valid, w_9)
            print("iter = %d, loss_5 = %f, loss_9 = %f" %(t, loss_5, loss_9))
            point_x.append(t)
            point_y[0].append(loss_5)
            point_y[1].append(loss_9)

    plt.plot( np.asarray(point_x), np.asarray(point_y[0]), 'c', label = 'w_5' )
    plt.plot( np.asarray(point_x), np.asarray(point_y[1]), 'g', label = 'w_9' )

    plt.title("rmse under different #features")
    plt.xlabel("number of iteration")
    plt.ylabel("loss rate (rmse)")
    # plt.xlim(0, iter_time)
    plt.ylim(0, 100)

    plt.legend()
    plt.savefig('prob2.png')
    plt.show()
    
    return w_5, w_9

if __name__ == '__main__':
    train_csv = sys.argv[1]
    np_file = sys.argv[2]
    
    x_train, x_valid, y_train, y_valid, mean_x, std_x = readData(train_csv)
    w_5, w_9 = gradientDescent(x_train, x_valid, y_train, y_valid)

    np.savez(np_file, w = w_9, mean_x = mean_x, std_x = std_x)