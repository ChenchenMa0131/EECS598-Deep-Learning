#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:02:58 2019

@author: chenchenma
"""

#%%%%
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import numpy as np
from sklearn.utils import shuffle
import os
#%%%
os.chdir('/Users/chenchenma/Documents/OneDrive/19winter/EECS598/Homework2/code/data')
train_data = pd.read_csv('train.txt', sep="|", header=None)
test_data = pd.read_csv('test.txt', sep="|", header=None)
dev_data = pd.read_csv('dev.txt', sep="|", header=None)
train_data = shuffle(train_data)
test_data = shuffle(test_data)
dev_data = shuffle(dev_data)
unlabelled = pd.read_csv('unlabelled.txt', sep="|", header= None)
unlabelled = [unlabelled.iloc[i][0].split() for i in range(unlabelled.shape[0])]

#%%%
def data_preprocess(data):
    labels = []
    text = []

    for i in range(data.shape[0]):
        t = (data.iloc[i])[0].split()
        labels.append(int(t[0]))
        text.append(t[1:])
        
    return(labels, text)

#%%%%
train_label, train_text = data_preprocess(train_data)
test_label, test_text = data_preprocess(test_data)
dev_label, dev_text = data_preprocess(dev_data)
train_data = (train_label, train_text)
test_data = (test_label, test_text)
dev_data = (dev_label, dev_text)

#%%%%
def vocabulary(text):
    vocab = {}
    for sent in text:
        for word in sent:
            if word not in vocab:
                vocab[word] = len(vocab)
    vocab['null'] = len(vocab)
    return vocab

vocab = vocabulary(train_text)

#%%%
def train(train_data, dev_data, net, criterion, optimizer, batchsize, num_epoch, device):
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        text = train_data[1]
        label = train_data[0]
        text_batch = text[:batchsize]
        label_batch = label[:batchsize]
        i = 0
        while len(label_batch) > 0:
            label_batch = torch.FloatTensor(label_batch)
            label_batch = label_batch.view(-1,1)
            optimizer.zero_grad()
            outputs = net(text_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            i += 1
            text_batch = text[i*batchsize:(i+1)*batchsize]
            label_batch = label[i*batchsize:(i+1)*batchsize]            
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss/100, end-start))
                start = time.time()
                running_loss = 0.0
        train_acc = test(train_data, net, device)
        dev_acc = test(dev_data, net, device)
        print('Accuracy of the network on the train set: %f %%' % (train_acc))
        print('Accuracy of the network on the development set: %f %%' % (dev_acc))
    print('Finished Training')
    
#%%%
def test(data, net, device):
    labels = np.squeeze(np.array(data[0]))
    texts = data[1]
    with torch.no_grad():
        outputs = net(texts)
        predicted = np.squeeze(outputs.numpy())
        predicted = predicted > 0.5
        total = len(labels)
        correct = np.sum((predicted == labels))
    acc = 100 * correct / total 
    return acc.item()

#%%
def predict(data, net, device):
    texts = data
    with torch.no_grad():
        outputs = net(texts)
        predicted = outputs.numpy()
        predicted = np.squeeze(predicted > 0.5)
    return predicted


#%%%%%%%%% Q1 %%%%%%%%%%%
class BOWClassifier(nn.Module):
    def __init__(self, vocab):
        super(BOWClassifier, self).__init__()
        self.vocab_size = len(vocab)
        self.lin = nn.Linear(self.vocab_size, 1)
        
    def bow_vectors(self, text):
        vec = torch.zeros(len(text), len(vocab))
        for i,sentence in enumerate(text):
            for word in sentence:
                if word not in vocab:
                    vec[i,vocab['null']]+=1
                else:
                    vec[i, vocab[word]]+=1
        return vec
    
    def forward(self, text):
        x = self.bow_vectors(text)
        return torch.sigmoid(self.lin(x))

#%%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCELoss()
netBOW = BOWClassifier(vocab).to(device)
optimizer = optim.Adam(netBOW.parameters(), lr=0.01)
train(train_data, dev_data, netBOW, criterion, optimizer, 200,10, device)


#%%%%
test_accuracy_bow = test(test_data, netBOW, device)
print('Accuracy of BOW on the test set: %f %%' % (test_accuracy_bow))


#%%%
predict1 = predict(unlabelled, netBOW, device)
np.savetxt('/Users/chenchenma/Documents/OneDrive/19winter/EECS598/Homework2/code/prediction/predictions_q1.txt', predict1, newline='\n',fmt='%i')



#%%%%%%%%%% Q2 %%%%%%%%
      
def word2ind(text, vocab):
    w2i = []
    for sentence in text:
        tmp = []
        for word in sentence:
            if word in vocab:
                tmp.append(vocab[word])
            else:
                tmp.append(vocab['null'])
        w2i.append(tmp)
    return w2i

train_ind = word2ind(train_text, vocab)
train_data2 = (train_label, train_ind)

test_ind = word2ind(test_text, vocab)
test_data2 = (test_label, test_ind)

dev_ind = word2ind(dev_text, vocab)
dev_data2 = (dev_label, dev_ind)

unlabelled_ind = word2ind(unlabelled, vocab)

#%%%
class EmbedClassifier(nn.Module):
    def __init__(self, vocab):
        super(EmbedClassifier, self).__init__()
        self.vocab_size = len(vocab)
        self.embed = nn.Embedding(self.vocab_size, 100)
        self.lin = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
        
    def format_input(self, x):
        lengths = [len(s) for s in x]
        max_len = max(lengths)
        input_var = np.zeros((len(x), max_len))
        for i, s in enumerate(x):
            input_var[i,:lengths[i]] = s
        input_var = torch.LongTensor(input_var)
        lengths = torch.FloatTensor(lengths)
        return (input_var , lengths)     
        
    def forward(self, x):
        input_var, lengths = self.format_input(x)
        embed = self.embed(input_var)
        embed = torch.sum(embed, 1) / lengths.view(-1,1)
        return self.sigmoid(self.lin(embed))


#%%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCELoss()
netEmbed = EmbedClassifier(vocab).to(device)
optimizer2 = optim.Adam(netEmbed.parameters(), lr=0.001)
train(train_data2, dev_data2, netEmbed, criterion, optimizer2,100, 10, device)


#%%%
test_accuracy_Embed = test(test_data2, netEmbed, device)
print('Accuracy of Embed on the test set: %f %%' % (test_accuracy_Embed))

#%%%
predict2 = predict(unlabelled_ind, netEmbed, device)
np.savetxt('/Users/chenchenma/Documents/OneDrive/19winter/EECS598/Homework2/code/prediction/predictions_q2.txt', predict2, newline='\n',fmt='%i')



#%%%%%%%%%% Q3 %%%%%%%%%%

# In this part, to import pre-trained GloVe, 
# I used some codes from https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
import os
import bcolz
import pickle
os.chdir('/Users/chenchenma/Documents/OneDrive/19winter/EECS598/Homework2/code/glove')

vocab_glove = {}
vectors_glove = bcolz.carray(0, rootdir=f'glove.100d.dat', mode='w')

with open(f'glove.twitter.27B.100d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        if(len(line[1:]) == 100):
            word = line[0]
            vocab_glove[word] = len(vocab_glove)
            v = np.array(line[1:]).astype(np.float)
            vectors_glove.append(v)

           
#%%%
vectors_glove = bcolz.carray(vectors_glove[1:].reshape((1193513, 100)), rootdir=f'glove.100d.dat', mode='w')
vectors_glove.flush()
pickle.dump(vocab_glove, open(f'27B.100d_vocab.pkl', 'wb'))

#%%%%%
vectors_glove = bcolz.open(f'glove.100d.dat')[:]
vocab_glove = pickle.load(open(f'27B.100d_idx.pkl', 'rb'))
glove = {w: vectors_glove[vocab_glove[w]] for w in vocab_glove.keys()}

#%%%%
weights = np.zeros((len(vocab), 100))

for i, word in enumerate(vocab):
    try: 
        weights[i] = glove[word]
    except KeyError:
        weights[i] = np.random.normal(scale=0.6, size=(100, ))


#%%%%
class GloVeClassifier(nn.Module):
    def __init__(self, weights):
        super(GloVeClassifier, self).__init__()
        self.num_embedding, self.embedding_dim = weights.shape
        self.embed = nn.Embedding(self.num_embedding, self.embedding_dim)
        self.embed.weight.data.copy_(torch.from_numpy(weights))
        self.lin = nn.Linear(self.embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def format_input(self, x):
        lengths = [len(s) for s in x]
        max_len = max(lengths)
        input_var = np.zeros((len(x), max_len))
        for i, s in enumerate(x):
            input_var[i,:lengths[i]] = s
        input_var = torch.LongTensor(input_var)
        lengths = torch.FloatTensor(lengths)
        return (input_var , lengths)             
        
    def forward(self, x):
        input_var, lengths = self.format_input(x)
        embed = self.embed(input_var)
        embed = torch.sum(embed, 1) / lengths.view(-1,1)
        return self.sigmoid(self.lin(embed))

#%%%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCELoss()
netGloVe = GloVeClassifier(weights).to(device)
optimizer3 = optim.RMSprop(netGloVe.parameters(), lr=0.001)
train(train_data2, dev_data2, netGloVe, criterion, optimizer3,100, 10, device)


#%%%
test_accuracy_GloVe = test(test_data2, netGloVe, device)
print('Accuracy of GloVe on the test set: %f %%' % (test_accuracy_GloVe))


#%%%
predict3 = predict(unlabelled_ind, netGloVe, device)
np.savetxt('/Users/chenchenma/Documents/OneDrive/19winter/EECS598/Homework2/code/prediction/predictions_q3.txt', predict3, newline='\n',fmt='%i')




#%%%%%%%%% Q4 & Q5 %%%%%%

class RnnClassifier(nn.Module):
    def __init__(self, weights,cell_type = 'rnn'):
        super(RnnClassifier, self).__init__()
        self.num_embedding, self.embedding_dim = weights.shape
        self.cell_type = cell_type.lower()
        if self.cell_type == 'rnn':
            self.rnn_cell = nn.RNN
        elif self.cell_type == 'lstm':
            self.rnn_cell = nn.LSTM
        self.embed = nn.Embedding(self.num_embedding, self.embedding_dim)
        self.embed.weight.data.copy_(torch.from_numpy(weights))
        self.embed.weight.requires_grad = True
        self.rnn = self.rnn_cell(self.embedding_dim, self.embedding_dim, batch_first=True)
        self.lin = nn.Linear(self.embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def format_input(self, x):
        n = len(x)
        lengths = np.array([len(t) for t in x], dtype = np.int64)
        max_len = np.max(lengths)
        input_var = np.zeros([n, max_len])
        for i in range(len(x)):
            input_var[i,:lengths[i]] = np.array(x[i])
        sort_ind = np.argsort(-lengths)
        lengths = lengths[sort_ind]
        input_var = input_var[sort_ind]
        input_var = torch.LongTensor(input_var)
        rev_ind = np.argsort(sort_ind)
        return (input_var, lengths, rev_ind)

    def forward(self, x):  
        input_var, lengths, rev_ind = self.format_input(x)
        embedded = self.embed(input_var)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.cell_type== 'lstm':
            hidden, cell = hidden
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = output[rev_ind]
        hidden = hidden[:, rev_ind]
        feats = hidden[0]
        return self.sigmoid(self.lin(feats))

#%%%% RNN %%%%%%%%%%%%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCELoss()
netRNN = RnnClassifier(weights,'rnn').to(device)
optimizer4 = optim.Adam(netRNN.parameters(), lr=0.001)
train(train_data2, dev_data2, netRNN, criterion, optimizer4,100, 10, device)


#%%%
test_accuracy_RNN = test(test_data2, netRNN, device)
print('Accuracy of RNN on the test set: %f %%' % (test_accuracy_RNN))


#%%%
predict4 = predict(unlabelled_ind, netRNN, device)
np.savetxt('/Users/chenchenma/Documents/OneDrive/19winter/EECS598/Homework2/code/prediction/predictions_q4.txt', predict4, newline='\n',fmt='%i')


#%%%% LSTM %%%%%%%%%%%%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCELoss()
netLSTM = RnnClassifier(weights,'lstm').to(device)
optimizer5 = optim.Adam(netLSTM.parameters(), lr=0.001)
train(train_data2, dev_data2, netLSTM, criterion, optimizer5,100, 10, device)


#%%%
test_accuracy_LSTM = test(test_data2, netLSTM, device)
print('Accuracy of RNN on the test set: %f %%' % (test_accuracy_LSTM))


#%%%
predict5 = predict(unlabelled_ind, netLSTM, device)
np.savetxt('/Users/chenchenma/Documents/OneDrive/19winter/EECS598/Homework2/code/prediction/predictions_q5.txt', predict5, newline='\n',fmt='%i')
