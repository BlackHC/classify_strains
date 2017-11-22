#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:46:12 2017

@author: ajaver
"""

import tensorflow
import torch.nn as nn
import torch.nn.functional as F
import classify.models.model_resnet as model_resnet
import classify.models.model_w_embedding as mwe


if __name__ == '__main__':
    
    import os
    import sys
    import torch
    
    src_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    sys.path.append(src_dir)
    
    from classify.train import init_generator, Trainer, IS_CUDA

    log_dir_root = 'logs/'

    # if sys.platform == 'linux':
    #     log_dir_root = '/work/ajaver/classify_strains/results'
    # else:
    #     log_dir_root = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/'
    #

    params = dict(
            is_reduced = True,
            dataset = 'CeNDR',
            data_file = 'data/raw/CeNDR_skel_smoothed.hdf5', #give the path of the .hdf5 location,
        # otherwise it will use the defaults of my setup
            #_valid_strains = ['JU258', 'CB4856'] #use for testing
    )
    
    gen_details, train_generator, test_generator = init_generator(**params)
    
    model = model_resnet.ResNetS(model_resnet.Bottleneck,
                    [3, 4, 6, 3], 
                    n_channels=train_generator.n_channels, 
                    num_classes = train_generator.n_classes
                    )
    model_name = 'ResNet50'

    model = mwe.EmbeddingModel(len(train_generator.snps_data), 2048, model)
    
    criterion = mwe.FullLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    
    n_epochs = 200
    log_dir = '{}_{}'.format(model_name, gen_details)

    if IS_CUDA:
        print('This is CUDA!!!!')
        torch.backends.cudnn.benchmark = True #useful for arrays of fix dimension
        model = model.cuda()
        criterion = criterion.cuda()
    
    t = Trainer(model,
             optimizer,
             criterion,
             train_generator,
             test_generator,
             n_epochs,
             log_dir
             )
    t.fit()
