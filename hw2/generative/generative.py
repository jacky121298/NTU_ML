import sys
import numpy as np
import matplotlib.pyplot as plt

def normalize(X, train = True, sp_col = None, x_mean = None, x_std = None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data
    if sp_col == None:
        sp_col = np.arange( X.shape[1] )
    
    if train:
        x_mean = np.mean(X[:, sp_col], axis = 0).reshape(1, -1)
        x_std = np.std(X[:, sp_col], axis = 0).reshape(1, -1)

    X[:, sp_col] = (X[:, sp_col] - x_mean) / (x_std + 1e-8)
    return X, x_mean, x_std

def sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def func(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return sigmoid(np.matmul(X, w) + b)

def predict(X, w, b):
    # This function returns a truth value prediction for each row of X 
    # by rounding the result of logistic regression function.
    return np.round( func(X, w, b) ).astype(np.int)

def accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

if __name__ == '__main__':
    x_train_f = sys.argv[1]
    y_train_f = sys.argv[2]
    x_test_f = sys.argv[3]
    output_f = sys.argv[4]

    with open(x_train_f) as xf:
        next(xf)
        x_train = np.array([line.strip('\n').split(',')[1:] for line in xf], dtype = float)

    with open(y_train_f) as yf:
        next(yf)
        y_train = np.array([line.strip('\n').split(',')[1] for line in yf], dtype = float)

    with open(x_test_f) as xf:
        next(xf)
        x_test = np.array([line.strip('\n').split(',')[1:] for line in xf], dtype = float)

    x_train, x_mean, x_std = normalize(x_train, train = True)
    x_test, _, _ = normalize(x_test, train = False, sp_col = None, x_mean = x_mean, x_std = x_std)

    data_dim = x_train.shape[1]
    # Compute in-class mean
    x_train_0 = np.array([x for x, y in zip(x_train, y_train) if y == 0])
    x_train_1 = np.array([x for x, y in zip(x_train, y_train) if y == 1])

    mean_0 = np.mean(x_train_0, axis = 0)
    mean_1 = np.mean(x_train_1, axis = 0)

    # Compute in-class covariance
    cov_0 = np.zeros((data_dim, data_dim))
    cov_1 = np.zeros((data_dim, data_dim))

    for x in x_train_0:
        cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / x_train_0.shape[0]
    for x in x_train_1:
        cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / x_train_1.shape[0]

    # Shared covariance is taken as a weighted average of individual in-class covariance.
    cov = (cov_0 * x_train_0.shape[0] + cov_1 * x_train_1.shape[0]) / (x_train_0.shape[0] + x_train_1.shape[0])

    # Compute inverse of covariance matrix.
    # Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
    # Via SVD decomposition, one can get matrix inverse efficiently and accurately.
    u, s, v = np.linalg.svd(cov, full_matrices = False)
    inv = np.matmul(v.T * 1 / s, u.T)

    # Directly compute weights and bias
    w = np.dot(inv, mean_0 - mean_1)
    b =  (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1)) + np.log(float(x_train_0.shape[0]) / x_train_1.shape[0]) 

    # Compute accuracy on training set
    y_train_pred = 1 - predict(x_train, w, b)
    print('Training accuracy: {}'.format( accuracy(y_train_pred, y_train) ) )

    # Predict testing labels
    predictions = 1 - predict(x_test, w, b)
    with open(output_f, 'w') as of:
        of.write('id,label\n')
        for i, label in enumerate(predictions):
            of.write('{},{}\n'.format(i, label))