#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 11:50:35 2019

@author: chenchenma
"""
#%%%
from softmax import *
from solver import *

#%%%%
import pickle, gzip
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

#%%%%%%%%%%%
data = {
    'X_train': X,# training data
    'y_train': Y,# training labels
    'X_val': X_test,# validation data
    'y_val': Y_test,# validation labels
  }

model1 = SoftmaxClassifier(weight_scale = 0.01)#, hidden_dim = 10)
solver1 = Solver(model1, data,
                update_rule='sgd',
                optim_config={
                    'learning_rate': 1#1e-3#,
                },
                lr_decay=1,
                num_epochs=10, batch_size=40,#,
                print_every=1000)
solver1.train()

#%%%%%%%%%%%
hidden_list = [50, 100, 200, 500, 1000, 1500]
data = {
    'X_train': X,# training data
    'y_train': Y,# training labels
    'X_val': X_val,# validation data
    'y_val': Y_val,# validation labels
  }
test_acc = []
val_acc = []

for h in hidden_list:
    
    model2 = SoftmaxClassifier(weight_scale = 0.01, hidden_dim = h)
    solver2 = Solver(model2, data,
                update_rule='adam',
                optim_config={
                    'learning_rate': 0.011#1e-3#,
                },
                lr_decay=1,
                num_epochs=10, batch_size=100,#,
                print_every=1000)
    solver2.train()
    val_acc.append(solver2.best_val_acc)
    test_acc.append(solver2.check_accuracy(X_test, Y_test,batch_size=len(Y_test)))

test_acc1 = solver1.check_accuracy(X_test, Y_test,batch_size=len(Y_test))
print(test_acc1)
print(val_acc)
print(test_acc)