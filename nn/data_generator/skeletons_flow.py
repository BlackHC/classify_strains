#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 09:20:28 2017
@author: ajaver

#add modifications by Andreas

"""
import random
import time
import numpy as np
import pandas as pd
import tables
import torch

from skeletons_transform import get_skeleton_transform, check_valid_transform

#MAYBE I SHOULD MOVE THIS TO __init__
import os
IS_CUDA = torch.cuda.is_available()
if IS_CUDA:
    # to prevent opencv from initializing CUDA in workers
    torch.randn(8).cuda()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

class SkeletonsFlowBase():
    _n_classes = None
    
    def __init__(self,
                 n_batch = 1, 
                 data_file = '',
                 set_type = None,
                 min_num_experiments = 1,
                 valid_strains = None,
                 sample_size_seconds = 90.,
                 sample_frequency_s =1 / 10.,
                 transform_type = 'angles',
                 is_normalized = False,
                 is_torch = False
                 ):

        check_valid_transform(transform_type)
        
        self.n_batch = n_batch
        self.sample_size_seconds = sample_size_seconds
        self.sample_frequency_s = sample_frequency_s
        self.sample_size = int(round(sample_size_seconds / sample_frequency_s))
        self.data_file = data_file
        self.transform_type = transform_type
        self.is_normalized = is_normalized
        self.is_torch = is_torch
        
        with pd.HDFStore(self.data_file, 'r') as fid:
            skeletons_ranges = fid['/skeletons_groups']
            experiments_data = fid['/experiments_data']
            self.strain_codes = fid['/strains_codes']
            self.strain_codes.index = self.strain_codes['strain_id']

            # read SNP only valid in CeNDR
            if '/snps_data' in fid:
                self.snps_data = fid['/snps_data']

        # Join the experiments and skeletons groups tables
        # I must use pd.join  NOT pd.merge to keep the same indexes as skeletons groups
        # otherwise the '/index_groups' subdivision will break
        cols_to_use = experiments_data.columns.difference(
            skeletons_ranges.columns)
        experiments_data = experiments_data[cols_to_use]
        experiments_data = experiments_data.rename(
            columns={'id': 'experiment_id'})
        skeletons_ranges = skeletons_ranges.join(
            experiments_data.set_index('experiment_id'), on='experiment_id')
        skeletons_ranges = skeletons_ranges.join(
            self.strain_codes.set_index('strain'), on='strain')
        
        self.data = tables.File(self.data_file, 'r')

        if set_type is not None:
            assert set_type in ['train', 'test', 'val']
            # use previously calculated indexes to divide data in training, validation and test sets
            valid_indices = self.data.get_node('/index_groups/' + set_type)[:]
            skeletons_ranges = skeletons_ranges.loc[valid_indices]

        # filter data to contain only the valid strains given
        if valid_strains is not None:
            skeletons_ranges = skeletons_ranges[
                skeletons_ranges['strain'].isin(valid_strains)]

        # minimum number of experiments/videos per strain
        skeletons_ranges = skeletons_ranges.groupby('strain_id').filter(
            lambda x: len(x['experiment_id'].unique()) >= min_num_experiments)

        self.skeletons_ranges = skeletons_ranges
    
    def _transform(self, skeletons):
        return get_skeleton_transform(skeletons, 
                               transform_type = self.transform_type,
                               is_normalized=self.is_normalized)
    
    def _read_skeletons(self, ini_r, fin_r):
        # TODO: might be even faster to
        while True:
            try:
                # read data. I use a while to protect from fails of data
                skeletons = self.data.get_node('/skeletons_data')[
                            ini_r:fin_r + 1, :, :]
                break
            except KeyError:
                print(
                    'There was an error reading the file, I will try again...')
                time.sleep(1)
        return skeletons
    
    def _to_torch(self, X, Y):
        X = np.rollaxis(X, -1, 1) # the channel dimension must be the second one
        
        Xt = torch.from_numpy(X).float()
        Yt = torch.from_numpy(Y).long()
        
        if IS_CUDA:
            Xt = Xt.cuda()
            Yt = Yt.cuda()
            
        input_var = torch.autograd.Variable(Xt)
        target_var = torch.autograd.Variable(Yt)
        
        return input_var, target_var
    
    def _serve_chunk(self, chunks):
        D =  map(np.array, zip(*chunks))
        if self.is_torch:
            D = self._to_torch(*D)
        
        return D
    

    @property
    def num_skeletons(self):
        return self.skeletons_ranges.shape[0]

    @property
    def n_classes(self):
        # number of classes for the one-hot encoding
        if not self._n_classes:
            self._n_classes = self.strain_codes['strain_id'].max() + 1 
        return self._n_classes
        
    def __len__(self):
        return self.num_skeletons // self.n_batch

class SkeletonsFlowShuffled(SkeletonsFlowBase):
    def __init__(self, **argkws):
        super().__init__(**argkws)
        
        
        # Only used when suffle == False.
        self.skeleton_id = -1
        
        # filter the chucks of continous skeletons to have at least the required sample size
        good = self.skeletons_ranges.apply(lambda x: x['fps'] * (
            x['fin'] - x['ini']) >= self.sample_size_seconds, axis=1)
        self.skeletons_ranges = self.skeletons_ranges[good]
        self.skel_range_grouped = self.skeletons_ranges.groupby('strain_id')
        self.strain_ids = list(map(int, self.skel_range_grouped.indices.keys()))
        
    def __next__(self):
        chunks = [self.next_single() for n in range(self.n_batch)]
        skeletons, strain_ids = self._serve_chunk(chunks)
        return skeletons, strain_ids
    
    
    
    def _random_choice(self):
        strain_id, = random.sample(self.strain_ids, 1)
        gg = self.skel_range_grouped.get_group(strain_id)
        ind, = random.sample(list(gg.index), 1)
        skeletons = self.prepare_skeleton(ind)

        return strain_id, skeletons
    
    def _random_transform(self, skeletons):
        # random rotation on the case of skeletons
        theta = random.uniform(-np.pi, np.pi)
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])

        skel_r = skeletons.copy()
        for ii in range(skel_r.shape[1]):
            skel_r[:, ii, :] = np.dot(rot_matrix, skeletons[:, ii, :].T).T

        # random mirrowing. It might be a problem since the skeletons do have left right orientation
        # for ii in range(skel_r.shape[-1]):
        #    skel_r[:, :, ii] *= random.choice([-1, 1])

        return skel_r
    
    def prepare_skeleton(self, ind):
        dat = self.skeletons_ranges.loc[ind]
        fps = dat['fps']

        # randomize the start
        sample_size_frames = int(round(self.sample_size_seconds * fps))
        r_f = dat['fin'] - sample_size_frames
        ini_r = random.randint(dat['ini'], r_f)
        fin_r = ini_r + sample_size_frames
        
        #read skeletons in blocks
        skeletons = self._read_skeletons(ini_r, fin_r)

        # get the expected row indexes
        row_indices = np.linspace(0, self.sample_size_seconds,
                                  self.sample_size) * fps
        row_indices = np.round(row_indices).astype(np.int32)
        skeletons = skeletons[row_indices]

        if np.any(np.isnan(skeletons)):
            print(ind, row_indices)
            # if there are nan we might have a bug... i am not sure how to solve it...
            raise ValueError('NaNs in skeletons data.')

        skeletons = self._transform(skeletons)
        return skeletons
    
    def next_single(self):
        strain_id, skeletons = self._random_choice()
        if self.transform_type == 'xy':
            skeletons = self._random_transform(skeletons)
            
        strain_id, skeletons = self._random_choice()
        if self.transform_type == 'xy':
            skeletons = self._random_transform(skeletons)
        return skeletons, strain_id

    
class SkeletonsFlowFull(SkeletonsFlowBase):
    _rows2iter = None
    def __init__(self, gap_btw_samples_s = None,  **argkws):
        super().__init__(**argkws)
        self.skeleton_id = -1
        
        if gap_btw_samples_s is None:
            gap_btw_samples_s = self.sample_size_seconds/2
        self.gap_btw_samples_s = gap_btw_samples_s

    
    
    def prepare_chunks(self, row):
        strain_id = row['strain_id']
        
        skeletons = self._read_skeletons(row['ini'], row['fin'])
        skeletons_t = self._transform(skeletons)
        
        dt = int(round(self.gap_btw_samples_s*row['fps']))
        fin = skeletons.shape[0] - self.sample_size
        
        chunks = [(skeletons_t[int(ini):int(ini) + self.sample_size], strain_id)
            for ini in range(0, fin, dt)]
        
        return chunks
    
    def __iter__(self):
        remainder = []
        for irow, row in self.skeletons_ranges.iterrows():
            chunks = self.prepare_chunks(row)
            chunks = remainder + chunks
            if len(chunks) >= self.n_batch:
                remainder = chunks[self.n_batch:]
                chunks = chunks[:self.n_batch]
                
                yield self._serve_chunk(chunks)
        
        if remainder:
            yield self._serve_chunk(remainder)
      
        
        