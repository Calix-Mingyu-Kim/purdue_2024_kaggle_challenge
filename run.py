# import cv2
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torchvision  import datasets, transforms
import torch.optim as optim
from torchvision.utils import make_grid
from torchvision.datasets.folder import default_loader
import torch.utils.data as data
from torchvision.models import resnet152, ResNet152_Weights, densenet121, DenseNet121_Weights, densenet201, DenseNet201_Weights

import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from loguru import logger

import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'resnet152'
lr = 8e-5
iter = 4
def plot_loss_curve(epochs, tr_loss, val_loss):
    epochs = np.arange(1, epochs + 1)
    plt.figure()
    plt.plot(epochs, tr_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and validation Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('plot/loss_curve_{}_{}_{}.png'.format(model_name, lr, iter))
    plt.close()
    
def plot_acc_curve(epochs, tr_acc, val_acc):
    epochs = np.arange(1, epochs + 1)
    plt.figure()
    plt.plot(epochs, tr_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('plot/accuracy_curve_{}_{}_{}.png'.format(model_name, lr, iter))
    plt.close()

class CelebFaceDataset(data.Dataset):
    def __init__(self, root, train=True, small=False, transform=None):
        self.root = root
        self.train = train
        self.small = small
        self.transform = transform
        self.images = os.listdir(self.root + 'train_small/') if small else os.listdir(self.root + 'train/') if train else os.listdir(self.root + 'test/')
        if train:
            self.labels = pd.read_csv(self.root + 'train_small.csv', index_col=False) if small else pd.read_csv(self.root + 'train.csv', index_col=False)
        self.class_dict = pd.read_csv(self.root + 'category.csv', index_col=False)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.root + 'train_small/' if self.small else (self.root + 'train/' if self.train else self.root + 'test/')
        if self.train:
            target = self.labels.iloc[idx]
            img = Image.open(img_path + target['File Name'])
        else:
            img = Image.open(img_path + '{}.jpg'.format(idx))
        
        img = img.convert('RGB')    

        if self.transform:
            img = self.transform(img)
        if self.train:
            label = self.class_dict.loc[self.class_dict['Category'] == target['Category']].index.item()
            return img, label, target['Category']
        else:
            return img

def main():
    epoch = 10
    class_dict = pd.read_csv('category.csv', index_col=False)
    
    weights = ResNet152_Weights.DEFAULT
    # weights = DenseNet201_Weights.DEFAULT
    preprocess = weights.transforms()
    transform = preprocess
    model = resnet152(weights=ResNet152_Weights.DEFAULT)
    # model = densenet201(weights=DenseNet201_Weights.DEFAULT)
    
    model.fc = nn.Sequential(
        # nn.Linear(model.fc.in_features, 1024),
        # nn.BatchNorm1d(1024),
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, len(class_dict))
        # nn.Linear(model.fc.in_features, 512),
        # nn.Linear(512, len(class_dict)),
    )
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()
    
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    full_dataset = CelebFaceDataset(root='/home/kim3118/run/', train=True, small=False, transform=transform)
    train_size = int(0.95 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # torch.manual_seed(2)
    
    train_dataset, val_dataset = data.random_split(full_dataset, [train_size, test_size])
    test_dataset = CelebFaceDataset(root='/home/kim3118/run/', train=False, small=False, transform=transform)
    
    train_loader = data.DataLoader(train_dataset, 
                                   batch_size=64, 
                                   shuffle=True)
    val_loader = data.DataLoader(val_dataset, 
                                 batch_size=64, 
                                 shuffle=False)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=64,
                                  shuffle=False)
    
    if not os.path.exists('plot'):
        os.makedirs('plot')
    logger.info(f'Start Training, lr: {lr}, iteration: {iter}')
        
    train_loss, validation_loss = [], []
    train_accuracy, validation_accuracy = [], []
    best_validation_acc = 0
    for ep in tqdm(range(epoch)):
        logger.info(f'Epoch {ep+1}/{epoch}')
        tr_loss = 0
        tr_acc = 0
        for batch in train_loader:
            imgs = batch[0]
            labels = batch[1]
            celebs = batch[2]
            
            imgs, labels = imgs.to(device), labels.to(device)
            model.train()
            pred = model(imgs)
            
            loss = loss_fn(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            
            pred = torch.log_softmax(pred, dim=1)
            _, predicted = torch.max(pred.data, 1)
            tr_acc += (predicted == labels).sum().item()
        
        tr_loss /= len(train_loader)
        tr_acc /= len(train_dataset)
        logger.info(f'Train Loss: {round(tr_loss,4)}, Train Accuracy: {round(tr_acc, 4)}')
        
        train_loss.append(round(tr_loss, 4))
        train_accuracy.append(tr_acc)
        
        with torch.no_grad():
            model.eval()
            val_loss = 0
            val_acc = 0
            logger.info('Validation')
            for batch in val_loader:
                imgs = batch[0]
                labels = batch[1]
                celebs = batch[2]
                
                imgs, labels = imgs.to(device), labels.to(device)
                pred = model(imgs)
                
                loss = loss_fn(pred, labels)
                val_loss += loss.item()
                pred = torch.log_softmax(pred, dim=1)
                _, predicted = torch.max(pred.data, 1)
                print("pd:", predicted)
                print("gt:", labels)
                val_acc += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            val_acc /= len(val_dataset)
            logger.info(f'Validation Loss: {round(val_loss,4)}, Validation Accuracy: {round(val_acc, 4)}')
            
            validation_loss.append(round(val_loss, 4))
            validation_accuracy.append(val_acc)
            
            if val_acc > best_validation_acc:
                best_validation_acc = val_acc
                torch.save(model.state_dict(), 'model_{}_{}_{}.pth'.format(model_name, lr, iter))
                test(model, class_dict, test_loader, False)
                logger.success("Updated csv!")
        scheduler.step()
        plot_loss_curve(ep+1, train_loss, validation_loss)
        plot_acc_curve(ep+1, train_accuracy, validation_accuracy)
        
    test_pred = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_loader):
            imgs = batch
            imgs = imgs.to(device)
            pred = model(imgs)
            pred = torch.log_softmax(pred, dim=1)
            _, predicted = torch.max(pred.data, 1)
            test_pred.append(predicted.cpu().numpy())
    print(test_pred)
    
# Testing
def test(model, class_dict, test_loader, called_from_main=True):
    if called_from_main:
        model.load_state_dict(torch.load('model_{}_{}_{}.pth'.format(model_name, lr, iter)))
    test_pred = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_loader):
            imgs = batch
            imgs = imgs.to(device)
            pred = model(imgs)
            pred = torch.log_softmax(pred, dim=1)
            _, predicted = torch.max(pred.data, 1)
            test_pred.append(predicted.cpu().numpy())
    print(test_pred)
    
    id = 0
    with open('test_pred_{}_{}_{}.csv'.format(model_name, lr, iter), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['Id','Category'])
        writer.writeheader()
        
        for pred in test_pred:
            for p in pred:
                writer.writerow({'Id':id, 'Category': class_dict.iloc[p]['Category']})
                id+=1
    

if __name__ == '__main__':
    main()
    # class_dict = pd.read_csv('category.csv', index_col=False)
    # model = resnet152(weights=ResNet152_Weights.DEFAULT)
    # weights = ResNet152_Weights.DEFAULT
    # preprocess = weights.transforms()
    # transform = preprocess
    # model.fc = nn.Sequential(
    #     nn.Dropout(0.5),
    #     nn.Linear(
    #         in_features=model.fc.in_features,
    #         out_features=len(class_dict)
    #     ),
    # )
    # model.load_state_dict(torch.load('model_resnet152.pth'))
    # model.to(device)
    # test_dataset = CelebFaceDataset(root='/home/kim3118/run/', train=False, small=False, transform=transform)
    
    # test_loader = data.DataLoader(test_dataset,
    #                               batch_size=64,
    #                               shuffle=False)
    # test(model, pd.read_csv('category.csv', index_col=False), test_loader, called_from_main=True)