

####################### Load Modules ###################################
import argparse
import netCDF4 as nc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import pandas as pd
import xarray as xr
import copy
import glob
from scipy import signal
from tqdm import tqdm
import logging

#from joblib import Parallel, delayed
#import multiprocessing

def compute_fss(selected_run_list = ['refrestart', 'radar'], date='20160606', thresholds=[0.1, 0.5, 1.], scales=[ 21, 41, 61, 81], prec_arrays = None): 
    """
    compute_fss loads in precipitation data (if not given) and then computes the fss. 
    
    Parameters
    ----------
    selected_run_list : list, optional
        has to indclude radar data, otherwise fss cannot be computed, by default ['refrestart', 'radar']
    date : str, optional
        [description], by default '20160606'
    thresholds : list, optional
        [description], by default [0.1, 0.5, 1.]
    scales : list, optional
        [description], by default [ 21, 41, 61, 81]
    
    Returns
    -------
    [pd.dataset]
        [Dataset contains FSS values for all thresholds, scales, times and runs]
    """
    # TODO: include ensemble dim
    from misc_functions import get_prec_data, get_radar_mask
    if not prec_arrays: 
        logging.info('Now reading in data.')
        prec_arrays = get_prec_data(run_names = selected_run_list,date=date)
        radar_mask = get_radar_mask(date=date, use_time_dependent=False)  
        prec_arrays = prec_arrays.where(radar_mask)
    list_t = []
    logging.info('Now computing fss')
    for t in tqdm(prec_arrays.time): 
        prec_sel = prec_arrays.sel(time=t)
        obs = prec_sel.radar.values
        list_run = []
        for run_name in selected_run_list[0:-1]:
            fcst = prec_sel[run_name].values
            egg = fss_frame(fcst, obs, windows=scales, levels = thresholds)[2]
            egg = egg.unstack().reset_index()
            egg = egg.rename(columns={'level_0':'scale', 'level_1': 'threshold', 0:'FSS'})
            egg['time']=t.values
            egg['run'] = run_name
            list_run.append(egg)
        list_t.append(pd.concat(list_run))
    fss_df = pd.concat(list_t)
    fss_df['FSS'] = fss_df.FSS.where(fss_df.FSS>0)
    return fss_df




# -*- coding: utf-8 -*-
#
#  K E N D A P Y . S C O R E _ F S S
#  compute Fractions (skill) score and related quantities
#
#  Almost completely adapted from
#  Faggian, Roux, Steinle, Ebert (2015) "Fast calculation of the fractions skill score"
#  MAUSAM, 66, 3, 457-466
#
#"""
#.. module:: score_fss
#:platform: Unix
#:synopsis: Compute the fraction skill score (2D).
#.. moduleauthor:: Nathan Faggian <n.faggian@bom.gov.au>
#"""


def compute_integral_table(field) :
    return field.cumsum(1).cumsum(0)

def fourier_filter(field, n) :
    return signal.fftconvolve(field, np.ones((n, n)))

def integral_filter(field, n, table=None) :
    """
    Fast summed area table version of the sliding accumulator.
    :param field: nd-array of binary hits/misses.
    :param n: window size.
    """
    w = n // 2
    if w < 1. :
        return field
    if table is None:
        table = compute_integral_table(field)

    r, c = np.mgrid[ 0:field.shape[0], 0:field.shape[1] ]
    r = r.astype(np.int)
    c = c.astype(np.int)
    w = np.int(w)
    r0, c0 = (np.clip(r - w, 0, field.shape[0] - 1), np.clip(c - w, 0, field.shape[1] - 1))
    r1, c1 = (np.clip(r + w, 0, field.shape[0] - 1), np.clip(c + w, 0, field.shape[1] - 1))
    integral_table = np.zeros(field.shape).astype(np.int64)
    integral_table += np.take(table, np.ravel_multi_index((r1, c1), field.shape))
    integral_table += np.take(table, np.ravel_multi_index((r0, c0), field.shape))
    integral_table -= np.take(table, np.ravel_multi_index((r0, c1), field.shape))
    integral_table -= np.take(table, np.ravel_multi_index((r1, c0), field.shape))
    # Kevin Bachmann: upper version adds zeros at the boundaries, the lower one wraps around (~periodic boundaries)
#    integral_table += np.take(table, np.ravel_multi_index((r1, c1), field.shape, mode='wrap'))
#    integral_table += np.take(table, np.ravel_multi_index((r0, c0), field.shape, mode='wrap'))
#    integral_table -= np.take(table, np.ravel_multi_index((r0, c1), field.shape, mode='wrap'))
#    integral_table -= np.take(table, np.ravel_multi_index((r1, c0), field.shape, mode='wrap'))
    return integral_table

def fourier_fss(fcst, obs, threshold, window) :
    """
    Compute the fraction skill score using convolution.
    :param fcst: nd-array, forecast field.
    :param obs: nd-array, observation field.
    :param window: integer, window size.
    :return: tuple of FSS numerator, denominator and score.
    """
    fhat = fourier_filter( fcst > threshold, window)
    ohat = fourier_filter( obs  > threshold, window)
    num = np.nanmean(np.power(fhat - ohat, 2))
    denom = np.nanmean(np.power(fhat, 2) + np.power(ohat, 2))
    return num, denom, 1.-num/denom

def fss(fcst, obs, threshold, window, fcst_cache=None, obs_cache=None):
    """
    Compute the fraction skill score using summed area tables .
    :param fcst: nd-array, forecast field.
    :param obs: nd-array, observation field.
    :param window: integer, window size.
    :return: tuple of FSS numerator, denominator and score.
    """
    fhat = integral_filter( fcst > threshold, window, fcst_cache )
    ohat = integral_filter( obs  > threshold, window, obs_cache  )

    num = np.nanmean(np.power(fhat - ohat, 2))
    denom = np.nanmean(np.power(fhat, 2) + np.power(ohat, 2))
    return num, denom, 1.-num/denom

def fss_frame(fcst, obs, windows, levels):
    """
    Compute the fraction skill score data-frame.
    :param fcst: nd-array, forecast field.
    :param obs: nd-array, observation field.
    :param window: list, window sizes.
    :param levels: list, threshold levels.
    :return: list, dataframes of the FSS: numerator,denominator and score.
    """
    num_data, den_data, fss_data = [], [], []
    for level in levels:
        ftable = compute_integral_table( fcst > level )
        otable = compute_integral_table( obs  > level )
        _data = [fss(fcst, obs, level, w, ftable, otable) for w in windows]
        num_data.append([x[0] for x in _data])
        den_data.append([x[1] for x in _data])
        fss_data.append([x[2] for x in _data])

    return ( pd.DataFrame(num_data, index=levels, columns=windows),
             pd.DataFrame(den_data, index=levels, columns=windows),
             pd.DataFrame(fss_data, index=levels, columns=windows))

### added by Leonhard Scheck 2018.1 : #########################################################################################

def fss_dict(fcst, obs, windows, levels) :
    """
    Compute the fraction skill score data-frame.
    :param fcst: nd-array, forecast field.
    :param obs: nd-array, observation field.
    :param window: list, window sizes.
    :param levels: list, threshold levels.
    :return: dictionary containting nd-array of the FSS: numerator,denominator and score (dim 0: levels, dim 1: windows).
    """
    num_data, den_data, fss_data = [], [], []
    for level in levels:
        ftable = compute_integral_table( fcst > level )
        otable = compute_integral_table( obs  > level )
        _data = [fss(fcst, obs, level, w, ftable, otable) for w in windows]
        num_data.append([x[0] for x in _data])
        den_data.append([x[1] for x in _data])
        fss_data.append([x[2] for x in _data])

    return { 'num':np.array(num_data), 'den':np.array(den_data), 'fss':np.array(fss_data), 'levels':levels, 'windows':windows }

def fss_ens_dict( ens, obs, windows, levels ) :
    """
    Apply fss_dict to all members of an ensemble, compute total fss.
    :param ens: nd-array, ensemble of forecast fields. Dimension 0 = ensemble dimension.
    :param obs: nd-array, observation field.
    :param window: list, window sizes.
    :param levels: list, threshold levels.
    :return: dictionary containting nd-array of the FSS: numerator,denominator and score + data for individual members.
    """

    #print 'fss_en_dict input : ', ens.shape, obs.shape, windows, levels

    fss_members = []
    for m in range(ens.shape[0]) :
        fss_members.append( fss_dict( ens[m,...], obs, windows, levels ) )

    #print '***', fss_members[0]['fss'].shape
    #print '+++', np.array( [ f['num'] for f in fss_members ] ).shape

    num =  np.array( [ f['num'] for f in fss_members ] ).sum(axis=0) # sum over ensemble dimension
    den =  np.array( [ f['den'] for f in fss_members ] ).sum(axis=0)
    fss = 1. - num/den

    #print '+++', fss.shape
    #raise ValueError('Hrmpf!')

    #print 'output shapes ', fss, fss_members[0]['fss']

    return { 'num':num, 'den':den, 'fss':fss, 'levels':levels, 'windows':windows, 'members':fss_members }

def fss_random_target( obs, levels ) :
    fss_random = np.count_nonzero( obs > levels ) / float(obs.size)
    fss_target = 0.5 + 0.5*fss_random
    return fss_random,  fss_target

