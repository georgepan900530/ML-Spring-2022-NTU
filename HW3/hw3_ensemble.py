_exp_name = "sample"
# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset, random_split,SubsetRandomSampler
from torchvision.datasets import DatasetFolder, VisionDataset
from torchvision import models
from sklearn.model_selection import KFold
# This is for the progress bar.
from tqdm.auto import tqdm
import random

myseed = 1126  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
    
from torchvision.transforms.transforms import RandomHorizontalFlip
# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
])

class FoodDataset(Dataset):

    def __init__(self,path1,tfm=test_tfm,files = None):
        super(FoodDataset).__init__()
        self.path1 = path1
        self.files = sorted([os.path.join(path1,x) for x in os.listdir(path1) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path1} sample",self.files[0])
        self.transform = tfm
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        #im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
        return im,label
    
# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

class Residual_Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, down_sample=None):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, 
                    out_channels=out_channel, 
                    kernel_size=3, 
                    stride=stride, 
                    padding=1,
                    bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel, 
                    out_channels=out_channel, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1,
                    bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.down_sample = down_sample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.down_sample:
            residual = self.down_sample(x)
        out += residual
        out = self.relu(out)

        return out
    
class Classifier(nn.Module):
    def __init__(self, block, layers, num_classes=11):
        super(Classifier, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.in_channels = 16
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=0.5)  # dropout
        
    def make_layer(self, block, out_channels, blocks, stride=1):  
        down_sample = None
        
        if (stride != 1) or (self.in_channels != out_channels):
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, down_sample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)  # 添加所有殘差塊

    def forward(self, x):
        out = self.conv(x) #(16,128,128)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer4(out)
        out = self.avg_pool(out)
        out = self.dropout(out) # dropout
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out
    
batch_size = 128
_dataset_dir = "/home/eegroup/eefrank/b08202036/ML/HW3/food11"
# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = FoodDataset(os.path.join(_dataset_dir,"training"), tfm=train_tfm)
valid_set = FoodDataset(os.path.join(_dataset_dir,"validation"), tfm=train_tfm)
full_set = ConcatDataset([train_set,valid_set])
full_loader = DataLoader(full_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# The number of training epochs and patience.
n_epochs = 100
patience = 30 # If no improvement in 'patience' epochs, early stop

# Initialize a model, and put it on the device specified.
model = Classifier(Residual_Block, [3, 3, 3, 3]).to(device)

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, verbose=True,min_lr=0.00001)
# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0

for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for batch in full_loader:

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()
        #print(imgs.shape,labels.shape)

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)
        
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    # update logs
    scheduler.step(train_acc)
    if train_acc > best_acc:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f} -> best")
    else:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")


    # save models
    if train_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
        best_acc = train_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break

model_best = Classifier(Residual_Block, [3, 3, 3, 3]).to(device)
model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))

def training(model,optimizer,criterion,dataloader):
    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

        # These are used to record information in training.
    train_loss = []
    train_accs = []

    for batch in dataloader:

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()
        #print(imgs.shape,labels.shape)

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)
            
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    return train_loss, train_acc
    
def validation(model,optimizer,criterion,dataloader):
    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in dataloader:

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        #break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    return valid_loss, valid_acc
    
def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
# The number of training epochs and patience.
n_epochs = 10
patience = 3 # If no improvement in 'patience' epochs, early stop
k = 5
splits=KFold(n_splits=k,shuffle=True,random_state=1126)
foldperf={}
history = {'train_loss': [], 'val_loss': [],'train_acc':[],'val_acc':[]}
# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
fold_acc = dict.fromkeys(['1','2','3','4','5'])
# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.0001, weight_decay=0.00001) 

# Initialize trackers, these are not parameters and should not be changed
for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(full_set)))):
    best_acc = 0
    stale = 0
    print(f'Fold {fold + 1}')
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(full_set, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(full_set, batch_size=batch_size, sampler=val_sampler)
    model.apply(reset_weights)
    for epoch in range(n_epochs):
        train_loss, train_acc = training(model_best,optimizer,criterion,train_loader)

        val_loss, val_acc = validation(model_best,optimizer,criterion,val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc.cpu())
        history['val_acc'].append(val_acc.cpu())
        
        # update logs
        if val_acc > best_acc:
            with open(f"./{_exp_name}_log.txt","a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {val_loss:.5f}, acc = {val_acc:.5f} -> best")
        else:
            with open(f"./{_exp_name}_log.txt","a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}")


        # save models
        if val_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model_best.state_dict(), f"{_exp_name}_Fold{fold+1}_best.ckpt") # only save best to prevent output memory exceed error
            best_acc = val_acc
            fold_acc[f'{fold+1}'] = best_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break
    foldperf[f'fold{fold+1}'] = history

testl_f,tl_f,testa_f,ta_f=[],[],[],[]
k=5
for f in range(1,k+1):

     tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
     testl_f.append(np.mean(foldperf['fold{}'.format(f)]['val_loss']))

     ta_f.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
     testa_f.append(np.mean(foldperf['fold{}'.format(f)]['val_acc']))

print('Performance of {} fold cross validation'.format(k))
print("Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test Acc: {:.2f}".format(np.mean(tl_f),np.mean(testl_f),np.mean(ta_f),np.mean(testa_f)))

model1 = Classifier(Residual_Block, [3, 3, 3, 3]).to(device)
model2 = Classifier(Residual_Block, [3, 3, 3, 3]).to(device)
model3 = Classifier(Residual_Block, [3, 3, 3, 3]).to(device)
model4 = Classifier(Residual_Block, [3, 3, 3, 3]).to(device)
model5 = Classifier(Residual_Block, [3, 3, 3, 3]).to(device)
model1.load_state_dict(torch.load(f"{_exp_name}_Fold1_best.ckpt"))
model2.load_state_dict(torch.load(f"{_exp_name}_Fold2_best.ckpt"))
model3.load_state_dict(torch.load(f"{_exp_name}_Fold3_best.ckpt"))
model4.load_state_dict(torch.load(f"{_exp_name}_Fold4_best.ckpt"))
model5.load_state_dict(torch.load(f"{_exp_name}_Fold5_best.ckpt"))
all_models = [model1.model2,model3,model4,model5]
preds = {'model1':[],'model2':[],'model3':[],'model4':[],'model5':[]}
for i in range(1,6):
    fold_model = all_models[i]
    fold_model.eval()
    with torch.no_grad():
        for data,_ in test_loader:
            test_pred = fold_model(data.to(device))
        for prob in test_pred:
            preds[f'model{i}'].append(prob)
prediction = []
for i in range(len(preds["model1"])):
    ensemble_pred = 0.2*preds["model1"][i] + 0.2*preds["model2"][i] + 0.2*preds["model3"][i] + 0.2*preds["model4"][i] + 0.2*preds["model5"][i]
    pred = ensemble_pred.argmax(0)
    prediction.append(pred.cpu().data.numpy())
    
#create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1,len(test_set)+1)]
df["Category"] = prediction
df.to_csv("/content/drive/MyDrive/ML/HW3/submission.csv",index = False)
