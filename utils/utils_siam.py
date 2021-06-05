##################################
########## Libraries #############
##################################

# Import torch
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Other Libraries
import random as rd
from PIL import Image
import PIL
import numpy as np
import matplotlib.pyplot as plt
import os
import random as rd

def get_cls_perf(test_dataloader, glob_dataloader, model, fs_class):
    '''
    Inputs:
        - test_dataloader (Dataloader) : dataloader for the test dataset
        - glob_dataloader (Dalaloader) : dataloader for a dataset containing one instance per classes
        - model (.pth): the siamese NN producing embeddingg
        - fs_class (int): the idx of the class you want to perform low shot learning on. This idx should correspond to the idx
            of the instance representing the class in the glob_dataset and  should represent the idx of the subfolder containing the 
            class instances in the test dataset (subfolders are sorted with alphanumerics)
    Outputs:
        -None
    ---------
    Prints the global accuracy and the Few-Shot accuracy
    '''
    # Getting performances
    perf, tot, perf_fs, tot_fs = 0, 0, 0, 0
    for i, data in enumerate(test_dataloader):
        scores = []

        img1, label = data
        img1, label = img1.cuda(), label.cuda()

        for j, data_glob in enumerate(glob_dataloader):
            img_g, label_g = data_glob
            img_g, label_g = img_g.cuda(), label_g.cuda()

            output1, output2 = model(img_g, img1)
            euclidean_distance = torch.pairwise_distance(output1, output2)
            scores.append(euclidean_distance)

            # If you want to see similarities
            if False:
                concatenated = torch.cat(
                    (img_g.cpu(), img1.cpu()), 0)
                imshow(torchvision.utils.make_grid(concatenated),
                            'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))

        pred_cls = np.argmin(scores)
        if pred_cls == label.item():
            perf += 1
        if label.item() == fs_class:
            tot_fs += 1
            if pred_cls == fs_class:
                perf_fs += 1
        tot += 1

    print("Precision is :", perf*100/tot, '%')
    print("Few-Shot precision is :", perf_fs*100/tot_fs, '%')



def get_glob(pathito, transform):
    '''
    Input: 
        pathito(str) -> the path to the folder containing the N instances representing the N classes of you dataset + one instance of the class
                        to few-shot
        transform (torch transform) -> the transform you want to apply to your image
    Ouput:
        glob (List) -> The list of pytorch tensors of those instances
    '''
    glob = []

    img_nm = os.listdir('./data/cifar100png/glob/')
    for i, name in enumerate(img_nm):
        img = Image.open(pathito + '/' + name)
        glob.append((transform(img).cuda(), i))

    return glob


# Inpired from https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch

class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform, one_channel=False, cls_dset=False, fs_class = None):
        self.imageFolderDataset = imageFolderDataset  # The image folder dataset
        self.transform = transform                    # The transform you want to apply to you images
        self.one_channel = one_channel                # If you images are one channel (mainly grayscale images)
        self.cls_dset = cls_dset
        self.fs_class = fs_class                      # If you want to build a dataset for classification purposes (only extracting one image with label)

    def __getitem__(self, index):
        if self.cls_dset:
            img_lbl = self.imageFolderDataset.imgs[index]
            img = Image.open(img_lbl[0])
            img = self.transform(img)
            label = img_lbl[1]

            return img, label

        else:
            img_lbl0 = rd.choice(self.imageFolderDataset.imgs)

            if self.fs_class > 0 and img_lbl0[1] != self.fs_class and rd.uniform(0,1) < 0.17:
                while img_lbl0[1] != self.fs_class:
                    img_lbl0 = rd.choice(self.imageFolderDataset.imgs)

            if rd.randint(0, 1):
                img_lbl1 = rd.choice(self.imageFolderDataset.imgs)
                while img_lbl0[1] != img_lbl1[1]:
                    img_lbl1 = rd.choice(self.imageFolderDataset.imgs)

            else:
                img_lbl1 = rd.choice(self.imageFolderDataset.imgs)
                while img_lbl0[1] == img_lbl1[1]:
                    img_lbl1 = rd.choice(self.imageFolderDataset.imgs)

            img0 = Image.open(img_lbl0[0])
            img1 = Image.open(img_lbl1[0])

            # If images are grayscale (one channel)
            if self.one_channel:
                img0 = img0.convert("L")
                img1 = img1.convert("L")

            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img_lbl1[1] != img_lbl0[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()
