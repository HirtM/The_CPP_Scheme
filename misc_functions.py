####################### Load Modules ###################################
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
#import cosmo_utils
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

from enstools.io import read
from enstools.plot import contour

def compute_theta_v(ta, pres, hus):
    p0 = 100000.  # reference pressure in PA
    kappa = 0.286  # R/cp =0.286 for air according to Wikipedia
    theta = ta * (p0 / pres) ** kappa
    r = hus * (1. - hus)  # mixing ratio of water vapor
    theta_v = theta * (1 + r / 0.622) / (1 + r)
    return theta_v

def get_prec_time(date='20160606', load_hourly_prec=True, run_name = 'reference'):
    radar_files = "/project/meteo/work/M.Hirt/radar_EY_de_domain/%s/*.nc"%(date)
    data_dir = '/project/meteo/scratch/M.Hirt/Cold_pool_perturbations/cosmo_runs/%s00/cde_%s_turlen500_%s00/OUTPUT/'%(date,run_name,date)
    if load_hourly_prec:
        cosmo_files = data_dir+'lfff*0000.nc_15min.nc'
    else: 
        logging.warning('Using 15 min data is not properly implemented yet!')
        cosmo_files = data_dir+'lfff*0000.nc_15min.nc'
    radar_mask = get_radar_mask(date=date, use_time_dependent=False)    
    ds_radar  = read(radar_files)
    ds_radar = ds_radar.where(radar_mask)
    prec = read(cosmo_files).TOT_PREC.where(radar_mask).diff('time')
    ds_radar['time'] = prec.time # overwrite to get same time 
    prec_radar= ds_radar.var61.sum(['rlon','rlat']).to_series()
    prec2 = prec.sum(['rlon','rlat']).to_series()
    df = pd.DataFrame()
    df['radar'] = prec_radar
    df['reference'] = prec2
    # use the following for seaborn format
    # df = df.unstack()
    # df.index.names=['data_name','time']
    # df.name = 'prec'
    # df = df.reset_index()
    return(df)



def get_radar_mask(date='20160606', product='EY', use_time_dependent=False):
    """
    get_radar_mask [summary]
    
    Parameters
    ----------
    date : str, optional
        [description], by default '20160606'
    product : str, optional
        [description], by default 'EY'
    use_time_dependent : bool, optional
        Whether mask is computed time independent, then only gridpoints where radarmask is always available is used, or if time dependent, mask will have dimensions rlon,rlat and time, by default False
    
    Returns
    -------
    [type]
        [description]
    """
    date = '20160606'
    radar_files = "/project/meteo/work/M.Hirt/radar_EY_de_domain/%s/*.nc"%(date)
    ds_radar  = read(radar_files)
    if use_time_dependent:
        radar_mask = ds_radar.var61.isnull()==False
    else:
        radar_mask = (ds_radar.var61.isnull()==False).min('time')
    radar_mask.name = 'radar_mask'
    radar_mask.attrs=dict(units='boolean', description='True, if radar data is available; False otherwise', data_files = radar_files )
    return radar_mask.compute()


def my_contour(xarray, lon='rlon', lat='rlat', **kwargs):
    """ This function makes a contour-map plot over germany for the cosmo
    output data. It is based on the ensemble tools contour function

    Parameter
    --------
    xarray (xarray data array):
        xarray dataarray that contains 2d fields with coordinate info
    **kwargs:
        see ensemble tools documentation of contour function for possible
        kwargs arguments.

    Returns
    -------

    """

    from enstools.plot import contour
    kwargs.setdefault('borders', '50m')
    kwargs.setdefault('coastlines', '50m')
    if xarray.min().values < 0 and xarray.max().values > 0:
        kwargs.setdefault('cmap', 'RdBu')
    else:
        kwargs.setdefault('cmap', 'viridis')

    #kwargs.setdefault('rotated_pole',dict(grid_north_pole_latitude=40, grid_north_pole_longitude=-170))
    try:
        kwargs.setdefault('rotated_pole', xarray.rotated_pole)
    except:
        pass
    #projection=ccrs.RotatedPole(pole_latitude=40, pole_longitude=-170)
    # kwargs.setdefault('transform',projection)
    fig,ax=contour(xarray, lon=xarray[lon], lat=xarray[lat], **kwargs)
    return fig,ax

def get_color_and_lw(selected_run_list, prec_arrays=None, radar_col = 'lightgrey', ref_col = 'dimgrey', cols=None):
    """
    get_color_and_lw assigns colors and linewidths to each run given in selected run list. Radar, reference runs and ensemble are treated specially.
    
    Parameters
    ----------
    selected_run_list : [list]
        [description]
    prec_arrays : [xr.dataset]
        required if ensembles are used --> deactivated currently
    radar_col : str, optional
        [description], by default 'lightgrey'
    ref_col : str, optional
        [description], by default 'dimgrey'
    other_cols : [type], optional
        [description], by default None
    """
    import matplotlib as mpl
    from cycler import cycle
    if not cols:
        cols = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    cols = cycle(cols)
    colors = []
    linewidths=[]
    for run in selected_run_list:
        if run =='radar': 
            colors.append(radar_col)
            linewidths.append(4)
        elif 'reference' in run:
            if '2mom' in run:
                colors.append(ref_col)
                linewidths.append(2)
            
            else:
                colors.append(ref_col)
                linewidths.append(4)
            #elif 'ens' in prec_arrays[run].dims:
        #    colors.append(next(cols))
        #    linewidths.append(1)  # required for ensembles
        else: 
            colors.append(next(cols))
            linewidths.append(2)
    return(colors, linewidths)

def get_prec_data(date='20160606', load_hourly_prec=True, run_names =['refrestart', 'cppertrestart_05','radar']):
    """
    get_prec_data loads precipitation data of runs given in run_names and properly computes hourly precipitation amount and makes it comparable to the also loaded radar data. 
    
    Parameters
    ----------
    date : str, optional
        [description], by default '20160606'
    load_hourly_prec : bool, optional
        not hourly does not work properly with radar data, by default True
    run_names : list, optional
        [description], by default ['refrestart', 'cppertrestart_05','radar']
    
    Returns
    -------
    xarray dataset
        dataset containing arrays of precipitation fields for each run 
    """
    radar_files = "/project/meteo/work/M.Hirt/radar_EY_de_domain/%s/*.nc"%(date)
    
    prec_arrays = xr.Dataset()
    radar_mask = get_radar_mask(date=date, use_time_dependent=False)   
    
    for run_name in tqdm(run_names):
        data_dir = '/project/meteo/scratch/M.Hirt/Cold_pool_perturbations/cosmo_runs/%s00/cde_%s_turlen500_%s00/*/'%(date,run_name,date)


        if len(glob.glob(data_dir))==0:
            data_dir = '/project/meteo/scratch/M.Hirt/Cold_pool_perturbations/cosmo_runs/%s00/cde_%s/*/'%(date,run_name)

#        if run_name =='PSP2': 
#           data_dir = '/project/meteo/scratch/M.Hirt/PSP-Schemes/COSMO_DE_Output/%s00/tur_len500/cde_PSP_TUR_newsettings_turlen500_%s00/OUTPUT/'%(date,date)
#            if date=='20160606':
#                data_dir = '/project/meteo/scratch/M.Hirt/PSP-Schemes/COSMO_DE_Output/%s00/tur_len500/cde_PSP-SH-w2uvHPBL_alpha15_%s00/OUTPUT/'%(date,date)
        if run_name =='reference_old': # this is the old reference run used for the PSP paper
            data_dir = '/project/meteo/scratch/M.Hirt/PSP-Schemes/COSMO_DE_Output/%s00/tur_len500/cde_reference_turlen500_%s00/OUTPUT/'%(date,date)

        if load_hourly_prec:
            cosmo_files = data_dir+'lfff*0000.nc_15min.nc'
        else: 
            logging.warning('Using 15 min data is not properly implemented yet!')
            cosmo_files = data_dir+'lfff*00.nc_5min.nc'
        if run_name =='radar': 
            prec  = read(radar_files).where(radar_mask).var61
            try: 
                prec['time'] = prec_arrays['refrestart'].time#[1:]
            except:     prec['time'] = prec_arrays['reference'].time#[1:]
            prec_arrays['radar']=prec
        else: 
            try:
                egg = read(cosmo_files,members_by_folder = True)
                egg = egg.TOT_PREC.compute()
                if len(egg.ens)==1: 
                    egg = egg.squeeze('ens')
                    del egg['ens']
                prec_arrays[run_name] = egg.diff('time',label='upper'  )
                #egg.differentiate('time',datetime_unit = 'h',  )
            except:
                logging.warning('directory %s does not exist, or something went wrong with reading in data. '%(data_dir))
            
            


    return prec_arrays



def initialize_map(ds, area='all', mask=None, ax=None):
    projection = ccrs.RotatedPole(
        pole_latitude=ds.rotated_pole.grid_north_pole_latitude,
        pole_longitude=ds.rotated_pole.grid_north_pole_longitude)
    ax = plt.axes(projection=projection)

    countries = cf.NaturalEarthFeature(
        category='cultural',
        name='admin_0_countries_lakes',
        scale='50m',
        facecolor='none')
    print(projection)

    ax.add_feature(countries, edgecolor='gray')
    if not mask is None:
        mask.values[np.isnan(mask.values)] = 2
        mask.plot.contourf(ax=ax, colors=('white', 'lightgray'), levels=np.array([0, 1.1, 2.1]),
                           extend='neither', transform=projection, x='rlon', y='rlat', add_colorbar=False)
    return(ax, projection)

def plot_prec_field(prec_dataset, run='ref', mask=None, savefig=True, 
                             figdir='', time=None, figdir_run='', cmap=None, levels=None):
    """
    This function plots a precipitation field for the specified subregions
    and the specified time steps. The e.g. hourly accumulated precipitation
    field is plotted.

    Parameters
    ----------
    prec_dataset:   xarray dataset object containing the precipitation fields
    mask:           radarmask with wicht to mask all precipitation fields
    ti (datetime ):
        whether to select single time steps

    Returns:
    --------
    """
    #    if figdir_run=='':
    #        figdir_run = figdir + run.name + '/'
    #    if not os.path.exists(figdir_run) and savefig:
    #        os.makedirs(figdir_run)

    if time:
        prec_dataset=prec_dataset.sel(time=[time])
    for ti, tstr in enumerate(prec_dataset.time):
        print(ti)
        ax, projection = initialize_map(prec_dataset, mask=mask)
        prec_dataset.isel(time=ti).plot.contourf(ax=ax, transform=projection, x='rlon', y='rlat',
                                                 add_labels=False, cmap=cmap,cbar=False,
                                                 cbar_kwargs={'label':'Precipitation [mm/h]'},
                                                 levels=levels,extend='max')
        ax.set_title(run,fontsize=20)
        #import pdb
        #pdb.set_trace()

        gl = ax.gridlines(color='gray', linestyle='--', alpha=0.5)
        hour=pd.to_datetime(tstr.values).strftime('%H')
        if savefig:
            figname = figdir_run + 'precipitation_' + \
                '_'.join(run) + '_' + hour + '.png'
            plt.savefig(figname,dpi=300)
            plt.clf()
        return ax


def plot_prec_fields(prec_arrays, 
    names =['radar', 'refrestart','cppertrestart_05']  , ti =16,
    radar_mask =None, 
    levels = np.array([0.1, 0.5, 1., 2., 4., 6., 10., 15, 20., 30, 40, 50, 70, 100]) ): 
    """
    plot_prec_fields [summary]
    
    Parameters
    ----------
    prec_arrays : [type]
        [description]
    names : list, optional
        [description], by default ['radar', 'refrestart','cppertrestart_05']
    ti : int, optional
        [description], by default 16
    """
    projection = ccrs.RotatedPole(
        pole_latitude=prec_arrays.rotated_pole.grid_north_pole_latitude,
        pole_longitude=prec_arrays.rotated_pole.grid_north_pole_longitude)
    countries = cf.NaturalEarthFeature(
        category='cultural',
        name='admin_0_countries_lakes',
        scale='50m',
        facecolor='none')
    gridlocations = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

    sfig_ncols = len(names)
    sfig_nrows = 1
    subfig_width=8

    precip_colormap = get_precip_colormap()
    

    fig, axs = plt.subplots(
            ncols=sfig_ncols, subplot_kw={'projection': projection},
            sharex=True, sharey=True,
            figsize=[subfig_width* sfig_ncols, 10* sfig_nrows + 0.5])

    for i, name in enumerate(names):
        axs[i].add_feature(countries, edgecolor='gray', linewidth=0.5)
        radar_mask.plot.contourf(ax=axs[i], colors=('lightgray', 'white'),alpha=.4,
                                            levels=np.array([0, .1, 2.1]),
                                            extend='neither', transform=projection,
                                            x='rlon', y='rlat', add_colorbar=False)

        try: 
            im = prec_arrays[name].isel(time=ti).plot.contourf(
                                ax=axs[i], transform=projection, x='rlon', y='rlat',
                                add_labels=False, cmap=precip_colormap, levels=levels,
                                extend='max', add_colorbar=False, zorder=10)
        except:
            logging.warning('Plotting of prec field for %s not possible. '%(name))

        mar=0.
        axs[i].set_extent([prec_arrays.rlon.min() - mar, prec_arrays.rlon.max() + mar,
                        prec_arrays.rlat.min() - mar, prec_arrays.rlat.max() +mar],
                        crs=projection)

        axs[i].set_title(name)

    fig.subplots_adjust(wspace=0.05)
    cb_kws=dict()
    cb_kws.setdefault('orientation', 'horizontal')
    cb_kws.setdefault('fraction', 0.1)
    cb_kws.setdefault('pad', 0.05)
    fig.colorbar(im, ax =axs.ravel().tolist(),**cb_kws)
    try: 
        fig.suptitle('Time=' + str(prec_arrays.time.isel(time=ti).dt.time.values), fontsize=20,)
    except: 
        fig.suptitle('Time=' + str(prec_arrays.time.isel(time=ti).values), fontsize=20,)
    return fig,axs


def get_precip_colormap():
    import matplotlib 
    # Colorbar with NSW Precip colors
    nws_precip_colors = [
    #"#FFFFFF",  # 0.0 - 0.01 inches
    "#04e9e7",  # 0.01 - 0.10 inches
    "#019ff4",  # 0.10 - 0.25 inches
    "#0300f4",  # 0.25 - 0.50 inches
    "#02fd02",  # 0.50 - 0.75 inches
    "#01c501",  # 0.75 - 1.00 inches
    "#008e00",  # 1.00 - 1.50 inches
    "#fdf802",  # 1.50 - 2.00 inches
    "#e5bc00",  # 2.00 - 2.50 inches
    "#fd9500",  # 2.50 - 3.00 inches
    "#fd0000",  # 3.00 - 4.00 inches
    "#d40000",  # 4.00 - 5.00 inches
    "#bc0000",  # 5.00 - 6.00 inches
    "#f800fd",  # 6.00 - 8.00 inches
    "#9854c6"  # 8.00 - 10.00 inches  
    ]
    precip_colormap = matplotlib.colors.ListedColormap(nws_precip_colors)
    return(precip_colormap)




def get_data(date='20160606', load_hourly=True, run_name ='cppertrestart_05'):
    """
    get_data loads in hourly data (not precipitation )
    
    Parameters
    ----------
    date : str, optional
        [description], by default '20160606'
    load_hourly : bool, optional
        [description], by default True
    run_name : str, optional
        [description], by default 'cppertrestart_05'
    
    Returns
    -------
    [type]
        [description]
    """
    
    data_dir = '/project/meteo/scratch/M.Hirt/Cold_pool_perturbations/cosmo_runs/%s00/cde_%s_turlen500_%s00/*/'%(date,run_name,date)
    if run_name =='PSP2': 
        data_dir = '/project/meteo/scratch/M.Hirt/PSP-Schemes/COSMO_DE_Output/%s00/tur_len500/cde_PSP_TUR_newsettings_turlen500_%s00/OUTPUT/'%(date,date)
    if len(glob.glob(data_dir))==0:
        data_dir = '/project/meteo/scratch/M.Hirt/Cold_pool_perturbations/cosmo_runs/%s00/cde_%s/*/'%(date,run_name)
    if load_hourly:
        cosmo_files = data_dir+'lfff*0000.nc_1h.nc'
    else: 
        logging.warning('Using other than hourly data is not properly implemented yet!')
        cosmo_files = data_dir+'lfff*_15min.nc'
    #logging.info(cosmo_files)
    dataset = read(cosmo_files)#.where(radar_mask)
    #logging.info(dataset)
#    dataset['level'] = dataset.level
#    dataset.set_coords('level')
    #dataset['level1'] = dataset.level
    #dataset.set_coords('level1')
    if load_hourly: 
        dataset['level'] = dataset.level
        dataset.set_coords('level')
        ds = dataset.isel(level=49)
        dataset['theta_v_sfc'] = compute_theta_v(ds['T'], ds.P, ds.QV).compute()

    return dataset



def plot_slice_T(dsets, sel_rlat=-1.5, itime =10):
    """
    plot_slice_T plots vertical slice along a rlat-line of theta-v field
    
    Parameters
    ----------
    dsets : [type]
        [description]
    sel_rlat : float, optional
        [description], by default -1.5
    itime : int, optional
        [description], by default 10
    """
    ds = dsets['cppert']
    if 'theta_v' not in ds:
        ds['theta_v'] = compute_theta_v(ds['T'], ds.P, ds.QV).compute()
    
    
    fig, axs = plt.subplots(nrows=3, sharex=True, sharey=False,
                                figsize=[22,9])
    time = dsets['cppert'].time.isel(time=itime)
    
    
    kws = dict( cmap='Spectral_r',figure=fig, ax = axs[0])
    dsets['ref']['theta_v_sfc'].sel(time=time, ).sel(rlat=slice(sel_rlat-.5,sel_rlat+.5)).plot(**kws)
    axs[0].hlines(y=sel_rlat,xmin=-5, xmax=5)
    axs[0].set_title('reference' )
    
    
    kws = dict( cmap='Spectral_r',figure=fig, ax = axs[1])
    ds['theta_v'].sel(time=time, level=49,).sel(rlat=slice(sel_rlat-.5,sel_rlat+.5)).plot(**kws)
    axs[1].hlines(y=sel_rlat,xmin=-5, xmax=5)
    axs[1].set_title(str(time.values)[0:16] )

    kws = dict(cmap='Spectral_r',figure=fig, ax=axs[2], levels = np.arange(296,303,0.2))
    ds['theta_v'].sel(time=time).sel(rlat = sel_rlat, method='nearest').plot(y = 'vcoord_ml',**kws)
    plt.ylim([0,3000])

    kws = dict(levels = np.array([-10.,-0.01,0.01,10.])/100000., colors='black',figure=fig, ax=axs[2])
    #ds.WTENS_CP.sel(time=time).sel(rlat = sel_rlat, method='nearest').plot.contour(y ='vcoord', **kws)
    ds.HPBL_SSO.sel(time=time).sel(rlat = sel_rlat, method='nearest').plot(figure=fig, ax=axs[2])


def compute_Iorg_2dfield(field, radar_mask = None, rlon=None, rlat = None,
                         boundaries_rlon=None, boundaries_rlat=None, 
                         rmax = 80000, dbin = 3000, 
                         return_all=False,
                         cld_args=dict(), cld_kws=dict()):
    """
    compute_Iorg_2dfield computes the organization Index I_org. 
    Notes:
    a) rmax should correspond at least to maximum nearest neighbour distance --> TODO: implement this
    b) Also the minimum, hence use a smaller dbin? 
    c) 
    
    Parameters
    ----------
    field : [2d]
        precipitation field 2d, dims[rlat,rlon]
    radar_mask : [2dfield], optional
        radar mask to apply, by default None
    rlon : [2d], optional
        2d meshgrid field or rlon, by default None
    rlat : [type], optional
        2d mesgrid field of rlat, by default None
    boundaries_rlon : [type], optional
        [description], by default None
    boundaries_rlat : [type], optional
        [description], by default None
    rmax : int, optional
        maximum radius for computing nearest-neighbour cdf (in m), by default 80000, but should probably be larger than 150m to capture all rmax
    dbin : int, optional
        delta-bin for nearest-neighbour cdf (in m), by default 3000
    return_all : bool, optional
        If True, also nncdf and nncdf_random are returned, otherwise only I_org
    cld_args : [dict], optional
        tuple of arguments that are passed to identify cloud function, by default dict()
    cld_kws : [dict], optional
        keyword-arguments that are passed to identify_cloud function, by default dict()

    Returns
    -------
    single value of I_org 
    """                
    from rdf_functions import identify_clouds         
    from scipy.spatial.distance import pdist, squareform
    dx = cld_kws['dx']
         
    # --------------- 1) identify clouds -------------------------------------
    labels, cld_size, cld_sum, com = identify_clouds(field,*cld_args, **cld_kws)
    
    # --------------- 2) compute nearest neighbour distances -----------------
    if com.size==0: # if empty fill with nans
        if return_all:
            return np.nan, np.nan, np.nan
        else:
            return np.nan
    dist = squareform(pdist(com) * dx)  # distance in m center to center
    com = (com * 0.025 - 5)  # transform to rlon/rlat
    np.fill_diagonal(dist, np.nan)  # remove zeros on diagonal (do not overwrite on dist --> is done automatically)
    nndist = np.nanmin(dist, 1)
    # ----------------- remove boundaries and radar mask ---------------------
    # remove values where no data is available (boundaries, radar mask

    min_dist_bdrd = []
    for cloud_id in range(0,len(com)) :
        # minimum distance to boundaries 
        min_dist_rlon = np.abs(
            (com[cloud_id,1] - boundaries_rlon)).min()
        min_dist_rlat = np.abs(
            (com[cloud_id,0]- boundaries_rlat)).min()
        min_dist_boundaries = np.min(
            [min_dist_rlat,
             min_dist_rlon])  # '/0.025  # in rlon/rlat units (degrees)
        # minimum distance to missing radar data
        dist_grid = np.sqrt(
            (com[cloud_id,1] - rlon) ** 2 + (
                    com[cloud_id,0] - rlat) ** 2)
        min_dist_radarmask = np.nanmin(np.where(radar_mask,np.nan, dist_grid))
        min_dist_bdrd.append( np.nanmin([min_dist_boundaries, min_dist_radarmask]) / 0.025 * dx)
    nndist = np.where(nndist<min_dist_bdrd, nndist, np.nan)

    # ------------- 3) compute I_org ------------------------------------------
    bins = np.arange(0, rmax, dbin)

    ds = xr.Dataset()
    # compute cumulative density function for nearest neighbour distances
    hist, bin_edges = np.histogram(nndist, bins=bins, density=True)
    nncdf = np.cumsum(hist) * dbin
    bins = bins[0:-1]

    # compute random reference
    # here we use not-nan values in com, becaus clouds with  nan values in nndist are included in other nns...
    l = np.nansum(np.isnan(com[:,0]) == False) / np.nansum(radar_mask)/dx/dx  # lambda determines poisson distribution
    nncdf_random = 1 - np.exp(-l * np.pi * bins ** 2)
    # compute area between diagonal and nncdf_random - nncdf curve
    I_org = np.trapz(nncdf, nncdf_random)
    if return_all:
        return I_org , nncdf, nncdf_random
    else:
        return I_org

# -----------------------------------------------------------------------------
# -------------------------- Main execution of Code ---------------------------
# -----------------------------------------------------------------------------    
if __name__=='__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s:%(message)s')
    
    print('Nothing to run... ')
