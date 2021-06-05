# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# %% Imports
import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

# Add
import utils.utils_siam as u_s
import random as rd
import Nets.siamese_net as n_s
import utils.my_criterion as crt
import matplotlib.pyplot as plt
import numpy as np

# Config


class cfg():
    # Path to the datasets
    training_dir = "./data/cifar100png/train/"
    testing_dir = "./data/cifar100png/test/"
    glob_dir = "./data/cifar100png/glob/"

    transform = transforms.Compose(
        [transforms.Resize((100, 100)), transforms.ToTensor()],)

    # Train and evaluate
    train = True
    evaluate = True

    # Hyper param
    train_batch_size = 64
    num_epochs = 10

    # Few-SHot class
    fs_class = 4


# Training dataset
folder_dataset = dset.ImageFolder(root=cfg.training_dir)

siamese_dataset = u_s.SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=cfg.transform,
                                            fs_class= cfg.fs_class)
train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,
                              num_workers=8,
                              batch_size=cfg.train_batch_size)

## NET ##
# Using ResNet18 or personnal net (uncomment if needed)
net = n_s.SiameseResNet().cuda()
# net = n_s.SiameseNetwork().cuda()


## LOSS ##
# Using Constrative loss with euclidean distance or cosine similarity (uncomment if needed)
criterion = crt.ContrastiveLoss()
# criterion = crt.ContrastiveLoss_cos()

# Optimizer
optimizer = optim.Adam(net.parameters(), lr=0.0005)


## TRAINING LOOP ##
if cfg.train:
    x_ax = []
    loss_history = []
    iteration_number = 0
    for epoch in range(cfg.num_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(
                    epoch, loss.item()))
                iteration_number += 10
                x_ax.append(iteration_number)
                loss_history.append(loss.item())
    plt.plot(x_ax, loss_history)
    plt.show()
    torch.save(net, './Nets/model.pth')


if cfg.evaluate:
    model = torch.load('./Nets/model.pth')
    model.eval()

    show_sim = False
    if show_sim:
        '''
        Pick a random image from the test dataset and compute dissmilarity with other ones. This is for visualization puporses only.
        Inpired from https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch (visualisation only)
        '''
        folder_dataset_test = dset.ImageFolder(root=cfg.testing_dir)
        siamese_dataset = u_s.SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                                    transform=cfg.transform
                                                    )
        test_dataloader = DataLoader(
            siamese_dataset, num_workers=2, batch_size=1, shuffle=True)
        dataiter = iter(test_dataloader)

        x0, _, _ = next(dataiter)

        for i in range(10):
            _, x1, label2 = next(dataiter)
            concatenated = torch.cat((x0, x1), 0)
            output1, output2 = model(x0.cuda(), x1.cuda())
            euclidean_distance = torch.pairwise_distance(output1, output2)
            u_s.imshow(torchvision.utils.make_grid(concatenated),
                       'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))

    show_cls = True
    if show_cls:
        '''
        Select one instances from train_set for each class. Then for each test sample, compute similarity with all those instances
        and label this sample according to the lowest dissimlarity (or highest similarity).
        '''
        ##########################
        # Loading the test dataset
        folder_dataset_test = dset.ImageFolder(root=cfg.testing_dir)
        siamese_dataset = u_s.SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                                    transform=cfg.transform,
                                                    cls_dset=True
                                                    )
        test_dataloader = DataLoader(
            siamese_dataset, num_workers=2, batch_size=1, shuffle=True)
        ##########################

        ##########################
        # Loading the glob instances
        folder_dataset_glob = dset.ImageFolder(root=cfg.glob_dir)
        glob_dataset = u_s.SiameseNetworkDataset(imageFolderDataset=folder_dataset_glob,
                                                 transform=cfg.transform,
                                                 cls_dset=True
                                                 )
        glob_dataloader = DataLoader(
            glob_dataset, num_workers=2, batch_size=1, shuffle=False)
        ##########################


        # Predict the class
        '''
        Get the instances with the lowest dissimilarity and classify according to this criteria
        '''
        fs_class = 4
        u_s.get_cls_perf(test_dataloader, glob_dataloader, model, fs_class)