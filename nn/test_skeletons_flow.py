#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:46:12 2017

@author: ajaver
"""
import sys
import os

from skeletons_flow import SkeletonsFlow

def _h_get_files(dataset):
    if sys.platform == 'linux':
        log_dir_root = '/work/ajaver/classify_strains/results'
        data_dir = os.environ['TMPDIR']
    else:        
        log_dir_root = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/logs/'
        data_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/train_data/'
        data_dir = os.path.join(data_dir, dataset)
        
    data_file =  os.path.join(data_dir, dataset + '_skel_smoothed.hdf5')
    
    return data_file, log_dir_root

def test_transforms(data_file):
    for tt in ['angles', 'eigenworms', 'eigenworms_full', 'xy']:
        for is_n in [False, True]:
            print(tt)
            gen = SkeletonsFlow(2, 
                          data_file, 
                          transform_type = tt,
                          is_normalized = is_n,
                          set_type = 'train', 
                          sample_size_frames_s = 10, 
                          sample_frequency_s=1/25.
                          )
            X,Y = next(gen)
            print(X.shape)
    
    
if __name__ == '__main__':

    #dataset = 'CeNDR'
    dataset = 'SWDB'
    
    data_file, log_dir_root = _h_get_files(dataset)
    test_transforms(data_file)
    
#%%
#for nn in range(X.shape[0]):
#    plt.figure()
#    plt.plot(X[nn, :, 0, 0])
#    plt.plot(X[nn, :, 25, 0])
#    plt.plot(X[nn, :, -1, 0])