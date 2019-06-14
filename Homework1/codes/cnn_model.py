#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 06:47:32 2019

@author: chenchenma
"""

from cnn import *
from solver import *
#%%%%
import pickle
import numpy as np

# Load the dataset
with open('mnist.pkl', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

#%%%%
X = np.array(list(train_set[0]) + list(valid_set[0]))
Y = np.concatenate([train_set[1], valid_set[1]])

#%%%%
X_test = test_set[0]
Y_test = test_set[1]

#%%%%
ind_val = np.random.choice(60000, 6000, replace = False)
ind_train = np.delete(np.arange(60000), ind_val)
#%%%%
X_val = X[ind_val]
Y_val = Y[ind_val]

#%%%
X_train = X[ind_train]
Y_train = Y[ind_train]

#%%%%
X_train_cnn = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test_cnn = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_val_cnn = X_val.reshape(X_val.shape[0], 1, 28, 28)

##%%%%
data = {
    'X_train': X_train_cnn,# training data
    'y_train': Y_train,# training labels
    'X_val': X_val_cnn,# validation data
    'y_val': Y_val,# validation labels
  }

model1 = ConvNet()#weight_scale = 0.01)#, hidden_dim = 10)
solver1 = Solver(model1, data,
                update_rule='adam',
                optim_config={
                    'learning_rate': 1e-3#,
                },
                lr_decay=0.95,
                num_epochs=5, batch_size=100,#,
                print_every=10)
solver1.train()
#%%%
test_acc = solver1.check_accuracy(X_test_cnn, Y_test,batch_size=len(Y_test))
print(test_acc)