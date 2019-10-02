import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches

model = []

def shuffle_data(data):
    "Randomly permute data pairs"
    
    p = np.random.permutation(len(data['t']))
    data_shf = {}
    data_shf['X'] = data_train['X'][p]
    data_shf['t'] = data_train['t'][p]


    return data_shf


def split_data(data, num_folds, fold):
    """ """ 
    x_split = np.array_split(data['X'], num_folds)
    y_split = np.array_split(data['t'], num_folds)

    data_fold = {'X': x_split[fold], 't': y_split[fold]}
    data_rest = {'X': np.vstack(x_split[:fold] + x_split[fold+1:]) , 't': np.concatenate(y_split[:fold] + y_split[fold+1:])}

    return [data_fold, data_rest]

def train_model(data, lambd):
    """wˆ MAP = (XX + λI)−1 X t """
    
    element1 = np.dot(data['X'].transpose(), data['X'])
    element2 = np.dot(lambd, np.identity(400))
    elementsum = np.add(element1, element2)
    elementinvert = np.linalg.inv(elementsum)
    w = np.dot(np.dot(elementinvert, data['X'].transpose()), data['t'])

    return w

def predict(data, model):
    """Predict target"""
    t = np.dot(data, model)
    return t

def loss(data, model): 
    """Calculate squared error"""
    error = abs(np.subtract(data['t'], np.dot(data['X'], model)))
    error = error.dot(error)/model.shape[0]
    return error

def cross_validation(data, num_folds, lambd_seq): 
    """ """
    cv_error = []
    data = shuffle_data(data)
    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0 
        for fold in range(num_folds):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        cv_error.append(cv_loss_lmd / num_folds)
    return cv_error



if __name__== "__main__":
    
    data_train = {'X': np.genfromtxt('data_train_X.csv', delimiter=','),
                't': np.genfromtxt('data_train_y.csv', delimiter=',')}
    data_test = {'X': np.genfromtxt('data_test_X.csv', delimiter=','),
                't': np.genfromtxt('data_test_y.csv', delimiter=',')}

    #data_shf = shuffle_data(data_train)
    #data_fold , data_rest = split_data(data_shf, 5, 2)
    #w = train_model(data_rest, lambd)
    
    lambd_seq = np.linspace(0.02, 1.5, 50)
    training_errors = []
    testing_errors = []

    cv_error5 = cross_validation(data_train, 5, lambd_seq)
    cv_error10 = cross_validation(data_train, 10, lambd_seq)

    for lambd in lambd_seq:
        model = train_model(data_train, lambd)
        training_errors.append(loss(data_train, model))
        testing_errors.append(loss(data_test, model))
    
    

    plt.plot(lambd_seq, training_errors, color = 'red', label = 'Train_Error')
    plt.plot(lambd_seq, testing_errors, color = 'orange', label = 'Testing_Error')
    plt.plot(lambd_seq, cv_error5, color = 'green', label = 'CV5_Error')
    plt.plot(lambd_seq, cv_error10, color = 'blue', label = 'CV10_Error')

    red_patch = mpatches.Patch(color='red', label='Train_Error')
    orange_patch = mpatches.Patch(color='orange', label='Testing_Error')
    green_patch = mpatches.Patch(color='green', label='CV5_Error')
    blue_patch = mpatches.Patch(color='blue', label='CV10_Error')
    plt.legend(handles=[red_patch, orange_patch, green_patch, blue_patch])
    plt.show()


  
