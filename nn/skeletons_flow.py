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
from skeletons_transform import get_skeleton_transform, check_valid_transform

# wild isolates used to test SWDB
SWDB_WILD_ISOLATES = ['JU393', 'ED3054', 'JU394',
                      'N2', 'JU440', 'ED3021', 'ED3017',
                      'JU438', 'JU298', 'JU345', 'RC301',
                      'AQ2947', 'ED3049',
                      'LSJ1', 'JU258', 'MY16',
                      'CB4852', 'CB4856', 'CB4853',
                      ]

# divergent set used for CeNDR
CeNDR_DIVERGENT_SET = ['N2', 'ED3017', 'CX11314', 'LKC34', 'MY16', 'DL238',
                       'JT11398', 'JU775',
                       'JU258', 'MY23', 'EG4725', 'CB4856']


class SkeletonsFlow():
    def __init__(self,
                 n_batch,
                 data_file,
                 set_type=None,
                 min_num_experiments=1,
                 valid_strains=None,
                 sample_size_frames_s=90,
                 sample_frequency_s=1 / 10,
                 transform_type = 'angles',
                 is_normalized = False,
                 shuffle=True
                 ):

        check_valid_transform(transform_type)

        self.n_batch = n_batch
        self.sample_size_frames_s = sample_size_frames_s
        self.sample_frequency_s = sample_frequency_s
        self.n_samples = int(round(sample_size_frames_s / sample_frequency_s))
        self.data_file = data_file
        self.transform_type = transform_type
        self.is_normalized = is_normalized
        self.shuffle = shuffle
        # Only used when suffle == False.
        self.skeleton_id = -1
       
        
        with pd.HDFStore(self.data_file, 'r') as fid:
            skeletons_ranges = fid['/skeletons_groups']
            experiments_data = fid['/experiments_data']
            self.strain_codes = fid['/strains_codes']
            self.strain_codes.index = self.strain_codes['strain_id']

            # read SNP only valid in CeNDR
            if '/snps_data' in fid:
                self.snps_data = fid['/snps_data']

        # number of classes for the one-hot encoding
        self.n_classes = self.strain_codes['strain_id'].max() + 1

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

        # filter the chucks of continous skeletons to have at least the required sample size
        good = skeletons_ranges.apply(lambda x: x['fps'] * (
            x['fin'] - x['ini']) >= self.sample_size_frames_s, axis=1)
        skeletons_ranges = skeletons_ranges[good]

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
        self.skel_range_grouped = skeletons_ranges.groupby('strain_id')
        self.strain_ids = list(map(int, self.skel_range_grouped.indices.keys()))

    def prepare_skeleton(self, ind):
        dat = self.skeletons_ranges.loc[ind]
        fps = dat['fps']

        # randomize the start
        sample_size_frames = int(round(self.sample_size_frames_s * fps))
        r_f = dat['fin'] - sample_size_frames
        ini_r = random.randint(dat['ini'], r_f)

        # TODO: might be even faster to
        while True:
            try:
                # read data. I use a while to protect from fails of data
                skeletons = self.data.get_node('/skeletons_data')[
                            ini_r:ini_r + sample_size_frames + 1, :, :]
                break
            except KeyError:
                print(
                    'There was an error reading the file, I will try again...')
                time.sleep(1)

        # get the expected row indexes
        row_indices = np.linspace(0, self.sample_size_frames_s,
                                  self.n_samples) * fps
        row_indices = np.round(row_indices).astype(np.int32)
        skeletons = skeletons[row_indices]

        if np.any(np.isnan(skeletons)):
            print(ind, row_indices)
            # if there are nan we might have a bug... i am not sure how to solve it...
            raise ValueError('NaNs in skeletons data.')

        skeletons = get_skeleton_transform(skeletons, 
                                           transform_type = self.transform_type,
                                           is_normalized=self.is_normalized)
        
        return skeletons

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

    def next_single(self):
        if self.shuffle:
            strain_id, skeletons = self._random_choice()
            if self.transform_type == 'xy':
                skeletons = self._random_transform(skeletons)
        else:
            self.skeleton_id += 1
            if self.skeleton_id >= self.num_skeletons:
                raise StopIteration()

            strain_id = self.skeletons_ranges.loc[self.skeleton_id].strain_id
            skeletons = self.prepare_skeleton(self.skeleton_id)

        # strain_one_hot = np.zeros(self.n_classes, np.int32)
        # strain_one_hot[strain_id] = 1

        return skeletons, strain_id

    def __iter__(self):
        self.skeleton_id = -1
        return self

    def __next__(self):
        D = [self.next_single() for n in range(self.n_batch)]
        skeletons, strain_ids = map(np.array, zip(*D))
        return skeletons, strain_ids

    @property
    def num_skeletons(self):
        return self.skeletons_ranges.shape[0]

    def __len__(self):
        return self.num_skeletons // self.n_batch
    
