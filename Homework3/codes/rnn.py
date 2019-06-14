import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import numpy as np
from sklearn.utils import shuffle
import os


os.chdir('/Users/chenchenma/Documents/OneDrive/19winter/EECS598/Homework2/code/data')
train_data = pd.read_csv('train.txt', sep="|", header=None)
test_data = pd.read_csv('test.txt', sep="|", header=None)
dev_data = pd.read_csv('dev.txt', sep="|", header=None)
train_data = shuffle(train_data)
test_data = shuffle(test_data)
dev_data = shuffle(dev_data)
unlabelled = pd.read_csv('unlabelled.txt', sep="|", header= None)
unlabelled = [unlabelled.iloc[i][0].split() for i in range(unlabelled.shape[0])]


def data_preprocess(data):
    labels = []
    text = []

    for i in range(data.shape[0]):
        t = (data.iloc[i])[0].split()
        labels.append(int(t[0]))
        text.append(t[1:])
        
    return(labels, text)

train_label, train_text = data_preprocess(train_data)
test_label, test_text = data_preprocess(test_data)
dev_label, dev_text = data_preprocess(dev_data)
train_data = (train_label, train_text)
test_data = (test_label, test_text)
dev_data = (dev_label, dev_text)


def vocabulary(text):
    vocab = {}
    for sent in text:
        for word in sent:
            if word not in vocab:
                vocab[word] = len(vocab)
    vocab['null'] = len(vocab)
    return vocab

vocab = vocabulary(train_text)


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

vectors_glove = bcolz.carray(vectors_glove[1:].reshape((1193513, 100)), rootdir=f'glove.100d.dat', mode='w')
vectors_glove.flush()
pickle.dump(vocab_glove, open(f'27B.100d_vocab.pkl', 'wb'))

vectors_glove = bcolz.open(f'glove.100d.dat')[:]
vocab_glove = pickle.load(open(f'27B.100d_idx.pkl', 'rb'))
glove = {w: vectors_glove[vocab_glove[w]] for w in vocab_glove.keys()}

weights = np.zeros((len(vocab), 100))


for i, word in enumerate(vocab):
    try: 
        weights[i] = glove[word]
    except KeyError:
        weights[i] = np.random.normal(scale=0.6, size=(100, ))



class CNN(nn.Module):
    def __init__(self, wegiths, kernel_size = 5, type = 'avg', num_filter = 128):
        super(CNN, self).__init__()

        self.num_embedding, self.embedding_dim = weights.shape
        self.kernel_size = kernel_size
        self.num_filter = num_filter

        self.embed = nn.Embedding(self.num_embedding, self.embedding_dim)
        self.embed.weight.data.copy_(torch.from_numpy(weights))

        self.conv = nn.Conv1d(self.embedding_dim,self.num_filter, kernel_size = self.kernel_size)
        self.GloAvgPool = nn.AdaptiveAvgPool1d(1)
        self.GloMaxPool = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()
        self.lin = nn.Linear(self.num_filter, 1)
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
        #embed = embed.view(embed.size(0),self.embedding_dim,-1)
        embed = embed.permute(0,2,1)
        h1 = self.conv(embed)
        if type == 'avg':
            h2 = self.GloAvgPool(h1).squeeze()
        elif type == 'max':
            h2 = self.GloMaxPool(h1).squeeze()
        h3 = self.relu(h2)
        output = self.sigmoid(self.lin(h3))     
        return output


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


def predict(data, net, device):
    texts = data
    with torch.no_grad():
        outputs = net(texts)
        predicted = outputs.numpy()
        predicted = np.squeeze(predicted > 0.5)
    return predicted



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCELoss()

test_acc_list = []
type_list = ['avg','max']
kernel_size = [5,7]
for type in type_list:
    for k in kernel_size:
        netCNN = CNN(weights, k, type).to(device)
        optimizer = optim.Adam(netCNN.parameters(), lr=0.0005)
        train(train_data2, dev_data2, netCNN, criterion, optimizer,20, 20, device)
        test_acc = test(test_data2, netCNN, device)
        test_acc_list.append(test_acc)
        print('Accuracy of CNN on the test set: %f %%' % (test_acc))

