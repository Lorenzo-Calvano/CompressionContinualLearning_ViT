import torchvision
import torch

from torchvision.io import read_image
from torchvision.models import *
from torch import nn
from collections import OrderedDict


img = read_image("C:\\Users\\ilcai\\Downloads\\automobile_cifar10.png")


# Step 1: Initialize model with the best available weights, then change it to desired output 
weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights)

heads: OrderedDict[str, nn.Linear] = OrderedDict()
heads["head"] = nn.Linear(model.hidden_dim, 10)
seq_heads = nn.Sequential(heads)
nn.init.zeros_(seq_heads.head.weight)
nn.init.zeros_(seq_heads.head.bias)
model.heads = seq_heads

#Initialize the inference transforms
preprocess = weights.transforms(antialias=True)

#load dataset and create dataloader (training, validation and test only)
training_set = torchvision.datasets.CIFAR10("Cifar10/train", transform=preprocess, train=True, download=True)
test_set = torchvision.datasets.CIFAR10("Cifar10/train", transform=preprocess, train=False, download=True)

train, valid =torch.utils.data.random_split(training_set, [45000,5000])

training_loader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
validation_loader = torch.utils.data.DataLoader(valid, batch_size=8, shuffle=False)


#set parameters for training/finetuning (to change)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
epochs = 3

#set gpu training for better performances
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

#function for accuracy
def accuracy_score(correct, total):
    return correct/total

#function to train one epoch
def train_one_epoch(epoch_index):
    
    #compute accuracy
    correct = 0
    instances = 45000
    
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        out = torch.argmax(outputs, dim=1)
        correct += ((out == labels).count_nonzero().item())
        
        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    #last batch (only 500 passes)
    last_loss = running_loss / 500
    print('  batch {} loss: {}'.format(45000, last_loss))

    
    return last_loss, accuracy_score(correct, instances)


'''
epoch_number = 0
best_vloss = 1_000_000.

for epoch in range(epochs):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss, accuracy_epoch = train_one_epoch(epoch_number)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    print(f"Accuracy score: {accuracy_epoch}")
    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}'.format(epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
'''

model.load_state_dict(torch.load("C:\\Users\\ilcai\\ViT_Params\\ViT_TorchVision\\model_2.bin", map_location=torch.device('cpu')), strict=False)


print(model.heads.head.weight)
img = preprocess(img)
a = model(img.unsqueeze(0)) 
print(a)