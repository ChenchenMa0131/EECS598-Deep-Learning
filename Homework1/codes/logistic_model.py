#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 21:20:44 2019

@author: chenchenma
"""
#%%
from solver import *
from logistic import *
#%%%
import os
import pickle

os.chdir('/Users/chenchenma/Documents/OneDrive/19winter/EECS598/hw1/deep-learning-course-master/Homeworks/Homework1/code')

with open('data.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
    
X = data[0]
Y = data[1]

X_train = X[:500]
X_val = X[500:750]
X_test = X[750:]

Y_train = Y[:500]
Y_val = Y[500:750]
Y_test = Y[750:]

#%%%%%%%%%%%
data = {
    'X_train': X_train,# training data
    'y_train': Y_train,# training labels
    'X_val': X_val,# validation data
    'y_val': Y_val,# validation labels
  }

model1 = LogisticClassifier(input_dim = 20, reg = 1e-3)#, hidden_dim = 10)
solver1 = Solver(model1, data,
                update_rule='sgd',
                optim_config={
                    'learning_rate': 1#1e-3#,
                },
                lr_decay=1,
                num_epochs=500, batch_size=20,#,
                print_every=100)
solver1.train()
test_acc1 = solver1.check_accuracy(X_test, Y_test,batch_size=len(Y_test))
print(test_acc1)

#%%% 2 layers
val_acc = []
test_acc = []

hidden_list = [50, 100, 200, 500, 1000]
for h in hidden_list :
    model2 = LogisticClassifier(input_dim = 20, 
                            hidden_dim = h, 
                            weight_scale = 0.01 ,
                            reg = 1e-4)#, hidden_dim = 10)
    solver2 = Solver(model2, data,
                update_rule='sgd',
                optim_config={
                    'learning_rate': 0.1#,
                },
                lr_decay=1,
                num_epochs=1000, batch_size=100,#,
                print_every=100)
    solver2.train()
    val_acc.append(solver2.best_val_acc)
    test_acc.append(solver2.check_accuracy(X_test, Y_test,batch_size=len(Y_test)))
print(val_acc)
print(test_acc)