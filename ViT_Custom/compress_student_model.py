import torch
from torch import nn
from ViT_Custom.ViT_Custom import ViT
from compress_mqa import tot_params
from torch.utils.data import DataLoader
import math

#loss function with both soft labels and true labels
class combined_loss(nn.Module):
    def __init__(self, loss1, loss2, alpha: float = 0.1):
        super().__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.alpha = alpha

    def forward(self, student_labels: torch.Tensor, teacher_labels: torch.Tensor, true_labels: torch.Tensor):
        #use true labels and pred_teacher for updating the loss function and optimize the student (combined loss)
        return self.loss1(true_labels, student_labels)*self.alpha + self.loss2(teacher_labels, student_labels)*(1-self.alpha) 



#compute validation dataset with teacher and student
def valid_student_run(teacher: ViT, student: ViT, valid_loader : DataLoader, loss_fn: combined_loss):
    tot_loss = 0

    for batch_idx, (inputs, true_labels) in enumerate(valid_loader):
        stud_labels = student(inputs)
        teach_labels = teacher(inputs)
        loss = loss_fn(stud_labels, teach_labels, true_labels)
        tot_loss += loss.item()
    
    return tot_loss/(batch_idx+1)



#function to train using knowledge distillation to let the student match the teacher's logits 
def train_student_one_epoch(teacher: ViT, student: ViT, training_loader: DataLoader, loss_fn: combined_loss, optimizer):
    tot_loss = 0

    for batch_idx, (inputs , true_labels) in enumerate(training_loader):
        pred_teacher = teacher(inputs)
        pred_student = student(inputs)

        loss = loss_fn(pred_student, pred_teacher, true_labels)
        tot_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return tot_loss/(batch_idx+1)



#train the student for tot epochs
def train_student(epochs: int, teacher: ViT, student: ViT, training_loader: DataLoader, validation_loader: DataLoader, loss_fn: combined_loss, optimizer):
    train_loss = 0
    min_loss = math.inf
    best = 0

    list_train_loss = []
    list_valid_loss = []
    for i in range(epochs):
        #train, compute the validation loss and print results
        train_loss = train_student_one_epoch(teacher, student, training_loader, loss_fn, optimizer)
        valid_loss = valid_student_run(teacher, student, validation_loader, loss_fn)
        print(f"Epoch number {i}: train loss --> {train_loss}, valid loss --> {valid_loss}")

        #save results for plotting purposes
        list_train_loss.append((i, train_loss))
        list_valid_loss.append((i, valid_loss))

        #save model only if better than before
        if valid_loss < min_loss:
            torch.save(student)
            min_loss = valid_loss
            best = i

    return best



#create teacher model and get the weights from pre-training
teacher = ViT()


#create a reduced ViT, with possibly less parameters: this will be the student model
student = ViT(num_blocks=8)

#set values for training the student
optimizer = torch.optim.Adam(params=student.parameters(), lr=0.001)
epochs = 10

#choose loss functions and combine them
loss1 = nn.CrossEntropyLoss()
loss2 = nn.CrossEntropyLoss()
loss_fn = combined_loss(loss1=loss1, loss2=loss2, alpha=0.2)


#load dataset for training and validation, then train to match the teacher

train_student(epochs, teacher, student, loss_fn=loss_fn, optimizer=optimizer)