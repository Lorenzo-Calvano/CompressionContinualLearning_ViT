import torch
import numpy as np
from ViT_Custom import ViT
from torch.utils.data.dataloader import DataLoader
from torchvision.models import *
import torchvision
import pandas as pd
import time
from flopth import flopth

#script to test the various ViT models, evaluate their performances after training 
#on inference speed, neural network dimension and accuracy scores
mb_size = 1024*1024


#scores for every class: precision, recall and f1-score
def cm_scores(confusion_matrix: pd.DataFrame):
    #compute recall, precision for class
    for column in confusion_matrix:
        col = confusion_matrix.iloc[:, column]
        row = confusion_matrix.iloc[column, :]
        recall = col[column]/(col.sum())
        precision = row[column]/(row.sum())
        if(precision !=0 and recall != 0):
            f_score = 2*(precision*recall)/(precision + recall)
        else:
            f_score = 0.0
        
        print(f"class {column}: recall --> {recall:.3f} ||| precision --> {precision:.3f} ||| f1-score --> {f_score:.3f}")
        
    return
        

#confusion matrix
def confusion_matrix(predictions, actual_labels):
    #done for confusion matrix simmetry, to remove when the ViT is trained
    predictions.append([0,1,2,3,4,5,6,7,8,9])
    actual_labels.append([9,8,7,6,5,4,3,2,1,0])

    #create pandas series
    predictions = np.array(predictions).flatten()
    actual_labels = np.array(actual_labels).flatten()
    pred = pd.Series(predictions, name="predicted")
    true = pd.Series(actual_labels, name="actual")

    confusion_matrix = pd.crosstab(pred, true)
    print(confusion_matrix)
    print()
    return confusion_matrix



#function to calculate the metrics  
def metrics_score(model : ViT, test_loader: DataLoader):
    correct = 0
    tot_time = 0
    actual_labels = []
    predictions = []

    for batch_idx, (inputs, true_labels) in enumerate(test_loader):

        start_time= time.perf_counter()
        pred_labels = model(inputs)
        end_time = time.perf_counter()
        tot_time += end_time - start_time

        out = torch.argmax(pred_labels, dim=1)
        correct += ((out == true_labels).count_nonzero().item())

        predictions.append(out)
        actual_labels.append(true_labels)

    
    print(f"mean inference time for batch size of {test_loader.batch_size} elements: {tot_time/(batch_idx+1)} seconds")
    print(f"accuracy value: {correct/((batch_idx+1)*test_loader.batch_size)}")
    print()
    print()

    #compute the confusion matrix, and accuracy/precision/recall/f1-score
    cm = confusion_matrix(predictions, actual_labels)
    cm_scores(cm)
    print()
    print()
    return



def return_type_size(torch_type):
    if(torch_type == torch.int8):
        return 1
    if(torch_type == torch.float16 or torch_type == torch.int16 or torch_type == torch.short):
        return 2
    if(torch_type == torch.float32 or torch_type == torch.int32):
        return 4
    if(torch_type == torch.float64 or torch_type == torch.int64  or torch_type == torch.double):
        return 8
    


#function to count number of parameters and size of network in MB(metric for compression)
def params_and_weight(model: ViT):
    tot_params = 0
    weight = 0
    for param in model.parameters():
        tot_params += param.flatten().shape[0]
        weight += param.flatten().shape[0]*return_type_size(param.flatten()[0].dtype)

    print(f"Numero dei parametri della rete: {tot_params}")
    print(f"Peso dei parametri della rete: {weight/mb_size} MB")
    return




def eval_model(model: ViT, test_loader: DataLoader, input_shape: tuple):
    #measure for weight and memory footprint
    params_and_weight(model)
    inp = torch.rand(input_shape)
    flops, params = flopth(model, inputs=(inp,), show_detail=True)
    print(flops, params)

    #measure for speed, accuracy, precision, recall etc...
    metrics_score(model, test_loader)
    return
    



#create Dataloader for test_set
weights = ViT_B_16_Weights.DEFAULT
preprocess = weights.transforms(antialias=True)
test_set= torchvision.datasets.CIFAR10("../Cifar10/train", transform=preprocess, train=False, download=True)
test, _ =torch.utils.data.random_split(test_set, [100,9900])
test_loader = DataLoader(test, batch_size=10, shuffle=False)



#model definition
model = ViT(num_blocks=3)

eval_model(model, test_loader, (10,3,224,224))