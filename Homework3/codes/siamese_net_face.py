import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
np.random.seed(498)
import random
random.seed(498)
from PIL import Image
import torch
torch.manual_seed(498)
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(65, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

class Config():
    training_dir = "att_faces/training/"
    testing_dir = "att_faces/testing/"
    train_batch_size = 64
    train_number_epochs = 200

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True,should_gray=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        self.should_gray = should_gray        

    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        
        if self.should_gray:
            img0 = img0.convert("L")
            img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        #TODO: design the architecture
        # Conv(in_channel=1, out_channel=8, kernel_size=3, padding=1)
        # MaxPool(kernel_size=2)
        # ReLU
        # BatchNorm
        # Conv(in_channel=8, out_channel=16, kernel_size=3, padding=1)
        # MaxPool(kernel_size=2)
        # ReLU
        # BatchNorm
        # Conv(in_channel=16, out_channel=32, kernel_size=3, padding=1)
        # MaxPool(kernel_size=2)
        # ReLU
        # BatchNorm

        # Fully connected(out_features=1024)
        # ReLU
        # Fully connected(out_features=512)
        # ReLU
        # Fully connected(out_features=8)

        self.conv1 = nn.Conv2d(1, 8, kernel_size = 3,padding = 1)
        self.max = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.batch1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size = 3, padding = 1)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)
        self.batch3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(4608, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 8)

    def forward_once(self, x):
        #TODO: implement the forward pass to get features for input image
        h1 = self.batch1(self.relu(self.max(self.conv1(x))))
        h2 = self.batch2(self.relu(self.max(self.conv2(h1))))
        h3 = self.batch3(self.relu(self.max(self.conv3(h2))))
        h3 = h3.view(h3.size(0), -1)
        h4 = self.relu(self.fc1(h3))
        h5 = self.relu(self.fc2(h4))
        output = self.fc3(h5)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        #TODO: argument output1 is f(x1), output2 is f(x2)
        # calculate the contrasive loss and return it
        # Note that in this function we have batch of samples as input.

        norm_diff = torch.norm(output1 - output2, dim = 1).view(-1,1)
        loss = (1-label) * norm_diff.pow(2) + label * (self.margin - norm_diff).clamp(min=0).pow(2)
        return loss.mean()

def evaluate(dataiter, net, split, device):
    for i in range(2):
        x0,_,_ = next(dataiter)
        for j in range(10):
            _,x1,_ = next(dataiter)
            concatenated = torch.cat((x0,x1),0)
            output1,output2 = net(Variable(x0).to(device),Variable(x1).to(device))
            euclidean_distance = F.pairwise_distance(output1, output2)
            imshow(torchvision.utils.make_grid(concatenated), '%s, dissimilarity:%.2f'%(split,euclidean_distance.item())) 
            plt.savefig('%s_%d_%d.png'%(split, i, j))
            plt.close()


if __name__ == "__main__":
    folder_dataset = dset.ImageFolder(root=Config.training_dir)
    
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()]),
                                        should_invert=False)
    train_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=10, batch_size=Config.train_batch_size)   
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0001)

    counter = []
    loss_history = [] 
    iteration_number= 0

    for epoch in range(0,Config.train_number_epochs):
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            img0, img1 , label = img0.to(device), img1.to(device) , label.to(device)
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 10 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
               
    train_dataloader = DataLoader(siamese_dataset, shuffle=False, num_workers=10, batch_size=1)
    dataiter = iter(train_dataloader)

    evaluate(dataiter, net, 'train', device)
    
    folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

    test_dataloader = DataLoader(siamese_dataset,num_workers=10,batch_size=1,shuffle=True)
    dataiter = iter(test_dataloader)
    evaluate(dataiter, net, 'test', device) 
  
    # Plot the training losses
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.show()

