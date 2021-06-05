#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 08:23:11 2021

@author: admin_loc
"""

import torch
from torch import autograd
from torch import nn


class ContrastiveLoss(nn.Module):
    # Taken from  https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive



class ContrastiveLoss_cos(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss_cos, self).__init__()
        self.margin = margin
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, output1, output2, label):
        cos_sim = self.cosine_sim(output1, output2) + 1
        loss_contrastive = torch.mean((1-label) * torch.pow(cos_sim, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - cos_sim, min=0.0), 2))


        return loss_contrastive