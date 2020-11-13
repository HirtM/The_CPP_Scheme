from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
# import cosmo_utils
import os
import pandas as pd
import xarray as xr
import copy
import glob
import cartopy.crs as ccrs
import cartopy.feature as cf
import seaborn as sns
import logging
from tqdm import tqdm
import fire
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import measurements
import warnings

warnings.filterwarnings("ignore")

from enstools.io import read
from enstools.plot import contour
from misc_functions import my_contour
from misc_functions import get_radar_mask
from misc_functions import get_prec_time
from misc_functions import compute_theta_v
from misc_functions import get_color_and_lw
from misc_functions import get_prec_data
from misc_functions import get_data, plot_slice_T
from rdf_functions import identify_clouds, calc_rdf


def load_whole_period(selected_run_list=['reference', 'radar'],
                      dates=['20160605', '20160606', '20160607']):
    ds_list = []
    rmask_list = []
    for date in dates:
        logging.info(date)
        prec_arrays = get_prec_data(run_names=selected_run_list, date=date)
        prec_arrays['start_date'] = prec_arrays.time.isel(
            time=0).time - np.timedelta64(1,
                                          'h')  # data starts at 1, not 00 --> subtract 1 hour for start_date
        prec_arrays['leadtime'] = (
                                              prec_arrays.time - prec_arrays.start_date) / np.timedelta64(
            1, 'h')
        prec_arrays = prec_arrays.set_coords(['leadtime', 'start_date'])
        prec_arrays = prec_arrays.swap_dims({'time': 'leadtime'})
        del prec_arrays['time']
        # prec_arrays['hour'] = prec_arrays.time

        ds_list.append(prec_arrays)
        radar_mask = get_radar_mask(date=date, use_time_dependent=False)
        rmask_list.append(radar_mask)
    prec_arrays = xr.concat(ds_list, dim='start_date')
    radar_mask = xr.concat(rmask_list, dim='time')
    radar_mask = radar_mask.rename({'time': 'start_date'})
    return prec_arrays, radar_mask


def identify_clouds_2(*args, return_var='labels', **kwargs):
    args = list(args)
    args[0] = args[0].squeeze()
    args = tuple(args)

    labels, cld_size, cld_sum = identify_clouds(*args, **kwargs)

    labels = xr.DataArray(labels, name='labels', dims=args[0].dims,
                          coords=args[0].coords)
    cld_size = xr.DataArray(cld_size, name='cloud_size', dims=dict(
        cloud_number=np.arange(0, len(cld_size))),
                            coords=dict(
                                cloud_number=np.arange(0, len(cld_size))))
    cld_sum = xr.DataArray(cld_sum, name='cloud_size', dims=dict(
        cloud_number=np.arange(0, len(cld_sum))),
                           coords=dict(
                               cloud_number=np.arange(0, len(cld_sum))))
    cloud_ds = xr.Dataset()
    cloud_ds['labels'] = labels
    cloud_ds['cloud_size'] = cld_size
    cloud_ds['cloud_sum'] = cld_sum
    cloud_ds['cloud_radius'] = np.sqrt(
        cloud_ds.cloud_size / np.pi) / 1000  # in km

    return (cloud_ds)


def compute_nncdf_iorg(nn_dist, radar_mask, dbin=3):
    # TODO: think about boundaries and radar mask!
    bins = np.arange(0, 80, dbin)
    bins_array = xr.DataArray(bins[1:], dims=['bins'],
                              coords={'bins': bins[1:]})
    ds = xr.Dataset()
    # compute cumulative density function for nearest neighbour distances
    hist, bin_edges = np.histogram(nn_dist, bins=bins, density=True)
    nncdf = np.cumsum(hist) * dbin
    ds['nncdf'] = xr.DataArray(nncdf, dims=['bins'],
                               coords={'bins': bins[1:]})
    # compute random reference
    l = (np.isnan(
        nn_dist) == False).sum() / radar_mask.sum()  # lambda determines poisson distribution
    nncdf_random = 1 - np.exp(-l * np.pi * bins_array ** 2)
    ds['nncdf_random'] = xr.DataArray(nncdf_random, dims=['bins'],
                                      coords={'bins': bins[1:]})

    # compute area between diagonal and nncdf_random - nncdf curve
    dx = ds.nncdf_random.diff('bins', label='lower')
    dy = ds.nncdf.diff('bins', label='lower')
    ds['I_org'] = (dx * dy * 0.5 + dx * ds.nncdf).sum('bins')  # -0.5  #
    return ds


def compute_Iorg(run_list, date, prec_thrs, water_kws, **kws):
    prec_arrays, radar_mask = load_whole_period(run_list, dates=[date])
    radar_mask['start_date'] = prec_arrays.start_date
    prec_arrays = prec_arrays.where(radar_mask)
    # ------------ identify clouds ------------------------------------------
    prec_stacked = prec_arrays.where(radar_mask).compute().to_array(
        dim='run', name='precipitation').stack(
        time_run=['leadtime', 'start_date', 'run'])

    I_org_ds = prec_stacked.groupby('time_run').apply(compute_Iorg_2dfield,
                                                      args=(
                                                      radar_mask, prec_thrs,),
                                                      **water_kws)
    I_org_ds = I_org_ds.unstack().rename_vars(
        {'time_run_level_0': 'leadtime', 'time_run_level_1': 'start_date',
         'time_run_level_2': 'run'})
    I_org_ds = I_org_ds.rename_dims(
        {'time_run_level_0': 'leadtime', 'time_run_level_1': 'start_date',
         'time_run_level_2': 'run'})

    return I_org_ds


def compute_Iorg_2dfield(prec_field, radar_mask, *cloud_args, **water_kws):
    logging.info(prec_field.time_run.values)
    cloud_ds = identify_clouds_2(prec_field, *cloud_args, **water_kws)
    # --------------- compute object distances ---------------------------------
    # center of mass + nearest neighbour
    n = cloud_ds.labels.max().values
    com = np.array(
        measurements.center_of_mass(prec_field.squeeze('time_run'),
                                    cloud_ds.labels, range(1, n)))

    dist = squareform(pdist(com) * 2.800)  # distance in km center to center
    com = (com * 0.025 - 5)  # transform to rlon/rlat
    com = xr.DataArray(com, dims=['cloud_id', 'com_dim'],
                       coords={'com_dim': np.array(['rlat', 'rlon']),
                               'cloud_id': range(1,
                                                 cloud_ds.labels.max().values)})
    # print(np.shape(dist))
    np.fill_diagonal(dist, np.nan)  # remove zeros on diagonal
    nndist = np.nanmin(dist, 1)
    nndist = xr.DataArray(nndist, dims=['cloud_id'],
                          coords={'cloud_id': range(1,
                                                    cloud_ds.labels.max().values)})
    # ----------------- remove boundaries and radar mask ------------------------
    # remove values where no data is available (boundaries, radar mask
    boundaries_rlon = [radar_mask.rlon.min().values,
                       radar_mask.rlon.max().values]
    boundaries_rlat = [radar_mask.rlat.min().values,
                       radar_mask.rlat.max().values]
    min_dist_bdrd = []
    for cloud_id, com_i in com.groupby('cloud_id'):
        min_dist_rlon = np.abs(
            (com_i.sel(com_dim='rlon').values - boundaries_rlon)).min()
        min_dist_rlat = np.abs(
            (com_i.sel(com_dim='rlat').values - boundaries_rlat)).min()

        min_dist_boundaries = np.min(
            [min_dist_rlat,
             min_dist_rlon])  # '/0.025  # in rlon/rlat units (degrees)

        dist_grid = np.sqrt(
            (com_i.sel(com_dim='rlon') - radar_mask.rlon) ** 2 + (
                    com_i.sel(com_dim='rlat') - radar_mask.rlat) ** 2)
        min_dist_radarmask = dist_grid.where(
            radar_mask == False).min().values
        min_dist_bdrd.append(
            min([min_dist_boundaries, min_dist_radarmask]) / 0.025 * 2.8)
    nndist = nndist.where(nndist < min_dist_bdrd)
    iorg_ds = compute_nncdf_iorg(nndist, radar_mask, dbin=3)
    return iorg_ds


def compute_cumRDF(run_list, date, prec_thrs, water_kws, rdf_kws, **kws):
    prec_arrays, radar_mask = load_whole_period(run_list, dates=[date])
    radar_mask['start_date'] = prec_arrays.start_date
    prec_arrays = prec_arrays.where(radar_mask)
    # ------------ identify clouds ------------------------------------------
    prec_stacked = prec_arrays.where(radar_mask).compute().to_array(
        dim='run', name='precipitation').stack(
        time_run=['leadtime', 'start_date', 'run'])
    count_list = []
    rdf_kws['dx'] = 2800
    rdf_kws['return_count_only'] = True
    from scipy.ndimage import uniform_filter
    convolved_mask = uniform_filter(radar_mask.squeeze(), rdf_kws['r_max'])
    for name, group in tqdm(prec_stacked.groupby('time_run')):
        prec_field = group.squeeze().values
        labels = identify_clouds(prec_field, prec_thrs, **water_kws)[0]
        count, radii, N_all, N_interior = calc_rdf(labels, prec_field, dr=1,
                                                   mask=convolved_mask,
                                                   **rdf_kws)
        count = xr.DataArray(count, dims=['radii'], coords={'radii': radii})
        ds = count.to_dataset(name='cell_count')
        ds['N_all'] = N_all
        ds['N_interior'] = N_interior
        egg = group.unstack('time_run')
        ds = ds.expand_dims(dict(leadtime=egg.leadtime,
                                 start_date=egg.start_date,
                                 run=egg.run))
        ds = ds.stack(time_run=['leadtime', 'start_date', 'run'])
        count_list.append(ds)
    count = xr.concat(count_list, 'time_run').unstack('time_run')
    return count



def integrate_prec(com_i, prec=None, radar_mask=None, rlon=None, rlat=None,
                   bins =None, normalize=True):
    rlon, rlat= np.meshgrid(rlon, rlat)
    dist_grid = np.sqrt((com_i[0] - rlat) ** 2 + (com_i[1] - rlon) ** 2)
    prec = np.where(prec>0.1,prec, 0) # small prec threshold
    idx = np.digitize(dist_grid,bins, right=False)
    prec_int = np.array([ np.mean(prec[idx==i]) for i in np.unique(idx)])
    # exclude where radarmask or boundaries are:
    m = radar_mask==False
    m[:,0] =True; m[:, -1] = True; m[0, :] = True;  m[-1, :] = True # include boundaries
    prec_int = np.where(bins < dist_grid[m].min(), prec_int, np.nan)
    if normalize:
        prec_int = prec_int/np.nanmean(np.where(radar_mask,prec, np.nan ))
        # normalize: output gives average prec. rate in the radius compared to
        # domain mean precipitation
    return prec_int


def compute_precRDF(run_list, date, prec_thrs, water_kws, **kws):
    prec_arrays, radar_mask = load_whole_period(run_list, dates=[date])
    radar_mask['start_date'] = prec_arrays.start_date
    prec_arrays = prec_arrays.where(radar_mask)
    # ------------ identify clouds ------------------------------------------
    prec_stacked = prec_arrays.where(radar_mask).compute().to_array(
        dim='run', name='precipitation').stack(
        time_run=['leadtime', 'start_date', 'run'])
    prec_int_list=[]
    for name, group in tqdm(prec_stacked.groupby('time_run')):
        prec_field = group.squeeze().values
        # identify cloud objects
        labels = identify_clouds(prec_field, prec_thrs, **water_kws)[0]
        # --------------- compute object centers of mass -----------------------
        # center of mass + nearest neighbour
        n = labels.max()#.values
        com = np.array(
            measurements.center_of_mass(prec_field, labels, range(1, n)))
        com = (com * 0.025 - 5)  # transform to rlon/rlat
        com = xr.DataArray(com, dims=['cloud_id', 'com_dim'],
                           coords={'com_dim': np.array(['rlat', 'rlon']),
                                   'cloud_id': range(1, labels.max())})

        #com.groupby('cloud_id').apply
        com = com.chunk({'cloud_id': 1})
        bins = np.arange(0, 2, 0.025) # bins in degree
        kwargs = dict(prec=group.squeeze().values,
                      radar_mask=radar_mask.mean('start_date').values,
                      rlon=group.rlon.values, rlat=group.rlat.values,
                      bins = bins)
        print(com.size)
        prec_int = xr.apply_ufunc(integrate_prec, com,
                              kwargs = kwargs,
                              input_core_dims=[['com_dim']],
                              output_core_dims=[['distance_bins']],
                              vectorize=True,
                              dask='parallelized', output_dtypes=[float],
                              output_sizes={'distance_bins': len(bins)})
        prec_int['distance_bins'] =bins/0.025*2.8 # in km

        egg = group.unstack('time_run')
        prec_int = prec_int.expand_dims(dict(leadtime=egg.leadtime,
                                 start_date=egg.start_date,
                                 run=egg.run))
        prec_int = prec_int.stack(time_run=['leadtime', 'start_date', 'run'])
        prec_int_list.append(prec_int.compute())
        #prec_int = prec_int.cumsum('distance_bins', skipna=False).mean(
        #    'cloud_id')
    return  xr.concat(prec_int_list, 'time_run').unstack('time_run')






def main(diagnostic='I_org', date='20160605'):
    data_dir = '/project/meteo/work/M.Hirt/CPP_data/precip_diagnostics_scores/'
    # -------------- Settings -------------------------------------------------
    water_kws = dict(water=True,
                     neighborhood=3)  # watershed segmentation for clouds
    settings = dict(water=water_kws['water'],
                    neighborhood=water_kws['neighborhood'],
                    prec_thrs=1.,
                    normalize_RDF=True,
                    rmax_RDF=30,  # in gridpoints = 80km
                    date_of_computation=str(datetime.today()),
                    )

    # -------------------------------------------------------------------------
    settings_str = 'prec_thrs_%.1f_' % (settings['prec_thrs'],)
    if water_kws['water']:
        settings_str = settings_str + 'watershed_separation_n_%i_' % (
            water_kws['neighborhood'])
    selected_run_list = ['reference',
                         'cppert09_a.2_theta1.5_sso50.0_H0500.0_HBPL1000.0_nfilter1',
                         'cppert09_a.2_theta1.5_sso50.0_H0500.0_HBPL500.0_nfilter1',
                         'PSP2', 'radar']
    if diagnostic == 'I_org':
        I_org_ds = compute_Iorg(run_list=selected_run_list, date=date,
                                prec_thrs=settings['prec_thrs'],
                                water_kws=water_kws)
        iorg_fname = data_dir + 'IORG_data_' + settings_str + '_' + date + '.nc'
        I_org_ds.compute().to_netcdf(iorg_fname, mode='w')
        print('I_org computation finished')
    if diagnostic == 'cumRDF':
        rdf_kws = dict(normalize=settings['normalize_RDF'],
                       r_max=settings['rmax_RDF'], )
        cumRDF = compute_cumRDF(run_list=selected_run_list, date=date,
                                prec_thrs=settings['prec_thrs'],
                                water_kws=water_kws, rdf_kws=rdf_kws)
        cumRDF_fname = data_dir + 'cumRDF_data_' + settings_str + '_' + date + '.nc'
        cumRDF.compute().to_netcdf(cumRDF_fname, mode='w')
        print('cumRDF computation finished')
    if diagnostic == 'precRDF':
        precRDF = compute_precRDF(run_list=selected_run_list, date=date,
                                  prec_thrs=settings['prec_thrs'],
                                  water_kws=water_kws)
        precRDF_fname = data_dir + 'prcRDF_data_' + settings_str + '_' + date + '.nc'
        precRDF.compute().to_netcdf(precRDF_fname, mode='w')
        print('precRDF computation finished')


# -----------------------------------------------------------------------------
# -------------------------- Main execution of Code ---------------------------
# -----------------------------------------------------------------------------    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s: %(levelname)s:%(message)s')
    # fire.Fire(main)
    dates = ['20160529', '20160530', '20160531', '20160601', '20160602',
             '20160603',
             '20160604', '20160605', '20160606', '20160607']
    for date in dates:
    # main(date=date)
        main(diagnostic='precRDF', date=date)
