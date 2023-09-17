import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score

from transformers import *
from datasets import load_dataset
from torchvision.transforms import *



#get the cifar10 dataset, and split in train, val e test (45k, 5k e 10k)
train_ds, test_ds = load_dataset("cifar10", split=['train[:50000]', 'test[:10000]'])
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

#preprocessing of images, both train and validation/test
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

#train preprocess step --> Crop image (random), Flip and normalization
normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

#validation (and test) preprocess step --> resize, crop at center and normalization
_val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )


#helper function to transform the images
def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
    return examples


#set the transformation to use for each split
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)



#dictionary for id and label of class, needed for setting the model
id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
label2id = {label:id for id,label in id2label.items()}

#get pretrained model ViT (on ImageNet21k) from hugging face, 
#last layer is MLP with as much output as needed for the classes of Cifar10
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                  id2label=id2label,
                                                  label2id=label2id)



#set the arguments for training (using hugging face API) the model, in our case it is used to fine tune
metric_name = "accuracy"

args = TrainingArguments(
    f"test-cifar-10",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,                  #learning rate decided
    per_device_train_batch_size=10,      #batch size
    per_device_eval_batch_size=4,
    num_train_epochs=3,                  #number of epoches
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,   #how to choose the best model of the various step
    logging_dir='logs',
    remove_unused_columns=False,
)


#metric computation, accuracy
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

#procedure to train
trainer.train()