#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .model_resnet import ResNetS, Bottleneck

__all__ = ['EmbeddingModel', 'FullLoss']


class EmbeddingModel(nn.Module):
    def __init__(self, snps_size, embedding_size, video_model):
        super(EmbeddingModel, self).__init__()
        self.video_model = video_model
        self.embedding_size = embedding_size
        self.snps_size = snps_size

        self.mapper = nn.Linear(snps_size, embedding_size)

    def forward(self, video_input, snps):
        classification, video_embedding = self.video_model(video_input)
        snps_embedding = self.mapper(snps)
        return classification, video_embedding, snps_embedding


class FullLoss(nn.Module):
    def __init__(self, embedding_loss_mixture=0.1):
        super().__init__()

        self.embedding_loss_mixture = embedding_loss_mixture
        self.classification_loss = nn.CrossEntropyLoss()
        self.embedding_loss = nn.MSELoss()

    def forward(self, embedding_output, target_classes):
        classification, video_embedding, snps_embedding = embedding_output
        classification_loss = self.classification_loss(classification,
                                                       target_classes)
        # Can't use the Loss layer here because it doesn't like
        embedding_loss = F.mse_loss(snps_embedding, video_embedding,
                                    size_average=True)
        loss = classification_loss + self.embedding_loss_mixture * \
                                     embedding_loss
        return loss

