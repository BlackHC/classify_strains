#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:44:21 2017

@author: ajaver
"""
import numpy as np
import warnings
import os


EIGENWORMS_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pca_components.npy')
EIGENWORMS_COMPONENTS = np.load(EIGENWORMS_FILE)

#Scaling factors obtained from the 2th, 50th, 98th percentiles of the transformed N2 data
EIGENWORM_SCALING_FACTOR = [[0., 25.], #delta_y
                        [0., 25.], #delta_x
                        [23., 7.5], #segment_size
                        [0., 0.04], #delta_angle
                        [0., 10.6], #eigen1
                        [0., 8.6], #eigen2
                        [0., 7.5], #eigen3
                        [0., 5.], #eigen4
                        [0., 2.75], #eigen5
                        [0., 1.5]] #eigen6

EIGENWORM_SCALING_FACTOR = np.array(EIGENWORM_SCALING_FACTOR)

ANGLE_SCALING_FACTOR = np.pi

#since the skeletons are centered with respect the body centroid this value changes more or less linearly
dd = np.linspace(100, 900, 25)
SKELETONS_SCALING_FACTOR = np.concatenate((dd[:0:-1], dd))


def _h_center_skeleton(skeletons, is_normalized, body_range = (8, 41)):
    body_coords = np.mean(skeletons[:, body_range[0]:body_range[1] + 1, :], axis=1)
    skeletons -= body_coords[:, None, :]
    
    if is_normalized:
        skeletons /= SKELETONS_SCALING_FACTOR[None, :, None]
    
    return skeletons

def _h_angles(skeletons, is_normalized):
    '''
    Get skeletons angles
    '''
    dd = np.diff(skeletons, axis=1)
    angles = np.arctan2(dd[..., 0], dd[..., 1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        angles = np.unwrap(angles, axis=1)

    mean_angles = np.mean(angles, axis=1)
    angles -= mean_angles[:, None]
    
    if is_normalized:
        angles /= ANGLE_SCALING_FACTOR

    return angles, mean_angles

def _h_eigenworms(skeletons, is_normalized, n_components = 6):
    angles, _ = _h_angles(skeletons, is_normalized=False)
    eigenworms = np.dot(angles, EIGENWORMS_COMPONENTS[:n_components].T)
    
    if is_normalized:
        eigenworms = (eigenworms-EIGENWORM_SCALING_FACTOR[4:,0])/EIGENWORM_SCALING_FACTOR[4:,1]
    return eigenworms

def _h_eigenworms_full(skeletons, is_normalized, n_components = 6):
    '''
    Fully transform the worm skeleton using its eigen components.
    
    The first four vectors are:
        (0,1) the change in head x,y position btw frame
        (2) the change in mean angle btw frames
        (3) each segment mean length
    
    The last six are the eigenvalues    
    
    '''
    angles, mean_angles = _h_angles(skeletons, is_normalized=False)
    eigenworms = np.dot(angles, EIGENWORMS_COMPONENTS[:n_components].T)
    
    mean_angles = np.unwrap(mean_angles)
    delta_ang = np.hstack((0, np.diff(mean_angles)))
    
    #get how much the head position changes over time but first rotate it to the skeletons to 
    #keep the same frame of coordinates as the mean_angles first position
    ang_m = mean_angles[0]
    R = np.array([[np.cos(ang_m), -np.sin(ang_m)], [np.sin(ang_m), np.cos(ang_m)]])
    head_r = skeletons[:, 0, :]
    head_r = np.dot(R, (head_r - head_r[0]).T)
    delta_xy = np.vstack((np.zeros((1,2)), np.diff(head_r.T, axis=0)))
    
    #size of each segment (the mean is a bit optional, at this point all the segment should be of equal size)
    segment_l = np.mean(np.linalg.norm(np.diff(skeletons, axis=1), axis=2), axis=1)
    
    #pack all the elments of the transform
    DT = np.hstack((delta_xy, segment_l[:, None], delta_ang[:, None], eigenworms))
    
    if is_normalized:
        DT = (DT-EIGENWORM_SCALING_FACTOR[:,0])/EIGENWORM_SCALING_FACTOR[:,1]
    
    return DT

def _h_eigenworms_T_inv(DT, is_normalized):
    '''
    Convert the eigen value transformed data into xy coordinates
    '''
    if is_normalized:
        DT = DT*EIGENWORM_SCALING_FACTOR[:,1] + EIGENWORM_SCALING_FACTOR[:,0]
    
    
    delta_y = DT[:, 0]
    delta_x = DT[:, 1]
    seg_l = DT[:, 2]
    delta_ang = DT[:, 3]
    
    eigenworms = DT[:, 4:]

    xx = np.cumsum(delta_x)
    yy = np.cumsum(delta_y)
    mean_angles = np.cumsum(delta_ang)
    
    n_components = eigenworms.shape[1]
    angles = np.dot(eigenworms, EIGENWORMS_COMPONENTS[:n_components])
    angles += mean_angles[:, None]
    
    
    ske_x = np.cos(angles)*seg_l[:, None]
    ske_x = np.hstack((xx[:, None],  ske_x))
    ske_x = np.cumsum(ske_x, axis=1) 
    
    ske_y = np.sin(angles)*seg_l[:, None]
    ske_y = np.hstack((yy[:, None],  ske_y))
    ske_y = np.cumsum(ske_y, axis=1) 
    
    skels_n = np.concatenate((ske_y[..., None], ske_x[..., None]), axis=1)
    
    return skels_n


def _angles(*args, **argkws):
    return _h_angles(*args, **argkws)[0]


#PULL ALL THE AVAILABLE DATA TRAMSFORMS TOGETHER
_skeletons_transforms = dict(
        xy = _h_center_skeleton,
        angles = _angles,
        eigenworms = _h_eigenworms,
        eigenworms_full = _h_eigenworms_full
        )

def check_valid_transform(transform_type):
    if not transform_type in _skeletons_transforms:
        dd = 'Only valid transforms are : {}'.format(_skeletons_transforms.keys())
        raise ValueError(dd)

def get_skeleton_transform(skeleton, transform_type,  is_normalized):
    dat = _skeletons_transforms[transform_type](skeleton, is_normalized=is_normalized)
    if dat.ndim == 2:
        dat = dat[..., None]
    return dat


    
    
    
    
