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

def train_dev(X, Y, ratio = 0.25):
    # This function spilts data into training set and development set.
    train_size = int( len(X) * (1 - ratio) )
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

def shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return X[randomize], Y[randomize]

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

def cross_entropy_loss(y_pred, Y_label):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy

def gradient(X, Y_label, w, b, lamb):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = func(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, axis = 1) + 2 * lamb * w
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad

if __name__ == '__main__':
    x_train_f = sys.argv[1]
    y_train_f = sys.argv[2]
    x_test_f = sys.argv[3]
    output_f = sys.argv[4]

    # To make the random numbers predictable
    np.random.seed(0)

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

    dev_ratio = 0.1
    x_train, y_train, x_dev, y_dev = train_dev(x_train, y_train, ratio = dev_ratio)

    # Regularization
    lamb = [0, 1, 5, 10, 50]
    lb = len(lamb)

    # Zero initialization for weights and bias
    data_dim = x_train.shape[1]
    w = np.zeros((data_dim, lb))
    b = np.zeros((1, lb))

    # Some parameters for training
    iter = 50
    batch = 10
    lr = 0.1

    # Keep the loss and accuracy at every iteration for plotting
    train_loss = [[] for _ in range(lb)]
    dev_loss = [[] for _ in range(lb)]
    train_acc = [[] for _ in range(lb)]
    dev_acc = [[] for _ in range(lb)]

    step = 1

    for epoch in range(iter):
        x_train, y_train = shuffle(x_train, y_train)
        # Mini-batch training
        for i in range( int( np.floor(len(x_train) / batch) ) ):
            X = x_train[i * batch : (i+1) * batch]
            Y = y_train[i * batch : (i+1) * batch]
            for l in range(lb):
                # Compute the gradient
                w_grad, b_grad = gradient(X, Y, w[:, l], b[:, l], lamb[l])
                # Learning rate decay with time
                w[:, l] = w[:, l] - lr / np.sqrt(step) * w_grad
                b[:, l] = b[:, l] - lr / np.sqrt(step) * b_grad
            
                step = step + 1
        
        print( 'Iteration : ' + str(epoch + 1) )

        for l in range(lb):
            # Compute loss and accuracy of training set and development set
            y_train_p = func(x_train, w[:, l], b[:, l])
            Y_train_p = np.round(y_train_p)
            train_acc[l].append( accuracy(Y_train_p, y_train) )
            train_loss[l].append( cross_entropy_loss(y_train_p, y_train) / len(y_train) )

            y_dev_p = func(x_dev, w[:, l], b[:, l])
            Y_dev_p = np.round(y_dev_p)
            dev_acc[l].append( accuracy(Y_dev_p, y_dev) )
            dev_loss[l].append( cross_entropy_loss(y_dev_p, y_dev) / len(y_dev) )

            print( 'lambda_{}'.format(lamb[l]) + ' : ' + str( accuracy(Y_dev_p, y_dev) ) )

    # Train accuracy curve
    legend = []
    for l in range(lb):
        plt.plot(train_acc[l])
        legend.append('lambda_' + str(lamb[l]))
    
    plt.title('Train Accuracy vs. lambda')
    plt.legend(legend)
    plt.savefig('train_acc.png')
    plt.show()

    # Development accuracy curve
    for l in range(lb):
        plt.plot(dev_acc[l])

    plt.title('Development Accuracy vs. lambda')
    plt.legend(legend)
    plt.savefig('dev_acc.png')
    plt.show()