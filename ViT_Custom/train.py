import torch
import torchvision
import math
from torch import nn
from torch.utils.data import DataLoader
from ViT_Custom import ViT
from torchvision.models import *



#function for single epoch
def train_one_epoch(model: ViT, train_loader : DataLoader, loss_fn, optimizer):
    tot_loss = 0
    for batch_idx, (inputs, true_labels) in enumerate(train_loader):
        #prediction from model on batch
        pred_labels = model(inputs)
        loss = loss_fn(pred_labels, true_labels)
        
        #keep track of total loss to compute the average loss during epoch
        tot_loss += loss.item()

        #gradient descent with optimizer algorithm
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print of batch loss (one every X batches)
        print(f"Number of batches: {batch_idx+1} done")
    
    #return average loss 
    return tot_loss/(batch_idx+1)
    

#valid run for computing the loss
def valid_run(model: ViT, valid_loader : DataLoader, loss_fn):
    tot_loss = 0

    for batch_idx, (inputs, true_labels) in enumerate(valid_loader):
        
        pred_labels = model(inputs)
        loss = loss_fn(pred_labels, true_labels)
        tot_loss += loss.item()
    
    return tot_loss/(batch_idx+1)



#function for the training loop
def train_model(model: ViT, loss_fn, optimizer, training_loader : DataLoader, validation_loader: DataLoader, epochs: int = 100):
    train_loss = 0
    min_loss = math.inf
    best = 0

    list_train_loss = []
    list_valid_loss = []
    for i in range(epochs):
        #train, compute the validation loss and print results
        print(f"Epoch number {i+1}:")

        train_loss = train_one_epoch(model, training_loader, loss_fn, optimizer)
        valid_loss = valid_run(model, validation_loader, loss_fn)
        print(f" End {i+1} Epoch: train loss --> {train_loss}, valid loss --> {valid_loss}")

        #save results for plotting purposes
        list_train_loss.append((i, train_loss))
        list_valid_loss.append((i, valid_loss))

        #save model only if better than before
        if valid_loss < min_loss:
            torch.save(model)
            min_loss = valid_loss
            best = i
    
    return best



#instance of the model
model = ViT()

#set values for training: num epochs, loss function, type of algorithm with parameters etc...
epochs = 100
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

#define the training set, validation set and test set with DataLoader
weights = ViT_B_16_Weights.DEFAULT
preprocess = weights.transforms(antialias=True)

#load dataset and create dataloader (training, validation and test only)
training_set = torchvision.datasets.CIFAR10("../Cifar10/train", transform=preprocess, train=True, download=True)
train, valid =torch.utils.data.random_split(training_set, [45000,5000])
training_loader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
validation_loader = torch.utils.data.DataLoader(valid, batch_size=8, shuffle=False)

train_model(model, loss_fn, optimizer, training_loader, validation_loader, epochs)

