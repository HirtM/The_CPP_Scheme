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
import fire
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import binary_erosion
from skimage import morphology
from enstools.io import read
from enstools.plot import contour

def detect_peaks(image, neighborhood = [[0,1,0],[1,1,1],[0,1,0]]):
    """
    This function is used in identify clouds and is not documented properly!!!
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    #neighborhood = generate_binary_structure(2,2)

    #neighborhood = np.ones((5, 5))

    #apply the local maximum filter; all pixel of maximal value
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to
    #successfully subtract it form local_max, otherwise a line will
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood,
                                       border_value=1)

    #we obtain the final mask, containing only peaks,
    #by removing the background from the local_max mask
    detected_peaks = local_max*1.-eroded_background*1.

    return detected_peaks

def identify_clouds(field, thresh, opt_field = None, opt_thresh = None,
                    water = False, dx = 2800., rho = None,
                    neighborhood=[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    return_com=False, return_cld_size_only=False):
    """
    copy from Stephan Rasp.

    Parameters
    ----------
    field : numpy.ndarray
      Field from which clouds are
    thresh : float
      Threshold for field
    opt_field : numpy.ndarray, optional
      Optional field used for creating a binary mask
    opt_thresh : float, optional
      Threshold for opt_field
    water : bool, optional
      If true, watershed algorithm is applied to identify clouds
    dx : float, optional
      Grid spacing [m]
    neighborhood : int or 2D numpy array
      Defines the search perimeter for cloud separation. Only valid of water is
      True
    return_com : bool
     If true, also returns list of centers of mass
    Returns
    -------
    labels : list
      List of labels
    cld_size : list
      List of cloud sizes
    cld_sum : list
      List of summed value of field for each cloud
    cof : np.array
      2D array with centers of mass, if return_com is True
    """
    from scipy.ndimage import measurements
 
    # Get binary field, 1s where there are clouds
    binfield = field > thresh
    if opt_field is not None:
        binfield *= opt_field > opt_thresh

    if water: # Apply watershed algorithm
        if type(neighborhood) is int:   # Convert integer to matrix
            neighborhood = np.ones((neighborhood, neighborhood))

        # Get local maxima
        lmax = detect_peaks(field*binfield, neighborhood=neighborhood)
        # Do the watershed segmentation
        # Get individual labels for local maxima
        lmax_labels, ncld = measurements.label(lmax)
        labels = morphology.watershed(-field, lmax_labels, mask = binfield)

    else:  # Regular algorithm
        # Find objects
        structure = [[0,1,0],[1,1,1],[0,1,0]]
        labels, ncld = measurements.label(binfield, structure = structure)
        #print('Regular ncld = %i' % (ncld))

    # Get sizes and sums
    cld_size = measurements.sum(binfield, labels, range(1, ncld+1))
    if rho is not None:
        field *= rho
    cld_sum = measurements.sum(field, labels, range(1, ncld+1))

    if return_com is not True:
        if return_cld_size_only: 
          return cld_size * dx * dx
        else:
          return labels, cld_size * dx * dx, cld_sum

    else:
        num = np.unique(labels).shape[0]  # Number of identified objects
        # Get centers of mass for each object
        cof = measurements.center_of_mass(field, labels, range(1, num))
        cof = np.array(cof)
        return labels, cld_size * dx * dx, cld_sum, cof


def identify_clouds_2(*args, return_var='labels',**kwargs):
    """ Wraps identify_clouds function and returns only the specified variable
    """
    args = list(args)
    args[0]=args[0].squeeze()
    args= tuple(args)
    labels, cld_size,cld_sum = identify_clouds(*args, **kwargs)
    if return_var =='labels': 
        labels = xr.DataArray(labels,name='labels',dims=args[0].dims, 
                               coords = args[0].coords )
        return(labels)
    elif return_var=='cld_size': 
        cld_size = xr.DataArray(cld_size,name='cloud_size',dims=dict(cloud_number=np.arange(0,len(cld_size))), 
                               coords = dict(cloud_number =np.arange(0,len(cld_size)) ))
        return(cld_size)
    elif return_var=='cld_sum': 
        cld_sum = xr.DataArray(cld_sum,name='cloud_size',dims=dict(cloud_number=np.arange(0,len(cld_sum))), 
                               coords = dict(cloud_number =np.arange(0,len(cld_sum)) ))
        return(cld_sum)

def calc_rdf(labels, field, normalize=True, dx=2800., r_max=30, dr=1, mask=None,
             return_count_only=False, **kws):
    """
    copy from SR.
    Computes radial distribution function
    Original credit : Julia Windmiller (MPI)
    
    Parameters
    ----------
    labels : numpy.ndarray
      Array with labels
    field : numpy.ndarray
      Original field Corresponding with labels field
    normalize : bool, optional
      If True normalize RDF
    dx : float, optional
      Grid spacing [m], used for r
    r_max : int, optional
      Maximum search radius for RDF algorithm (in grid pts)
    dr : int, optional
      Search step (in grid pts)
      
    Returns
    -------
    g: numpy.ndarray
      (Normalized) RDF
    r : numpy.ndarray
      Distance
    """
    from scipy.ndimage import measurements
    num = np.unique(labels).shape[0]   # Number of identified objects
    # Get centers of mass for each object
    cof = measurements.center_of_mass(field, labels, range(1,num))
    cof = np.array(cof)
    

    # If no centers of mass are found, an empty array is passed
    if cof.shape[0] == 0:   # Account for empty arrays
        cof = np.empty((0,2))
    cof = cof[np.isnan(cof[:, 0]) == False, :] # remove nan values
    # If no centers of mass are found, an empty array is passed
    if cof.shape[0] == 0:   # Account for empty arrays
        cof = np.empty((0,2))

    g, r, tmp = pair_correlation_2d(cof[:, 0], cof[:, 1],
                                    [field.shape[0], field.shape[1]],
                                    r_max, dr, normalize=normalize, mask=mask,
                                    return_count_only=return_count_only, **kws)
    if return_count_only:
        return g, r*dx#, len(cof), len(tmp) # N all clouds, N interior clouds
    else: return g, r*dx

def pair_correlation_2d(x, y, S, r_max, dr, normalize=True, mask=None,
                        return_count_only=False, reference_area=None):
    """
    copy from SR.
    Need new doc string 
    
    https://github.com/cfinch/colloid/blob/master/adsorption/analysis.py
    
    Compute the two-dimensional pair correlation function, also known
    as the radial distribution function, for a set of circular particles
    contained in a square region of a plane.  This simple function finds
    reference particles such that a circle of radius r_max drawn around the
    particle will fit entirely within the square, eliminating the need to
    compensate for edge effects.  If no such particles exist, an error is
    returned. Try a smaller r_max...or write some code to handle edge effects! ;)
    
    Arguments:
        x               an array of x positions of centers of particles
        y               an array of y positions of centers of particles
        S               length of each side of the square region of the plane
        r_max            outer diameter of largest annulus
        dr              increment for increasing radius of annulus
        reference_area  area (n-gridpoints) to be used for computing the total 
                        density (should not be convolved radar mask, but radarmask)
    Returns a tuple: (g, radii, interior_indices)
        g(r)            a numpy array containing the correlation function g(r)
        radii           a numpy array containing the radii of the
                        annuli used to compute g(r)
        reference_indices   indices of reference particles

    """

    # Number of particles in ring/area of ring/number of reference
    # particles/number density
    # area of ring = pi*(r_outer**2 - r_inner**2)

    # Extract domain size
    (Sx,Sy) = S if len(S) == 2 else (S, S)

    # Find particles which are close enough to the box center that a circle of radius
    # r_max will not cross any edge of the box

    # Find indices within boundaries
    if mask is None:
        bools1 = x > r_max
        bools2 = x < (Sx - r_max)
        bools3 = y > r_max
        bools4 = y < (Sy - r_max)
        interior_indices, = np.where(bools1 * bools2 * bools3 * bools4)
    else:
        # Get closes indices for parcels in a pretty non-pythonic way
        # and check whether it is inside convolved mask
        x_round = np.round(x)
        y_round = np.round(y)
        interior_indices = []
        for i in range(x_round.shape[0]):
            if not np.isnan(x_round[i]):
                if mask[int(x_round[i]), int(y_round[i])] == 1:
                    interior_indices.append(i)

    num_interior_particles = len(interior_indices)

    edges = np.arange(0., r_max + dr, dr)   # Was originally 1.1?
    num_increments = len(edges) - 1
    g = np.zeros([num_interior_particles, num_increments])
    radii = np.zeros(num_increments)
    if mask is None:
        number_density = float(len(x)) / float(Sx*Sy)
    else:
        number_density = float(len(x)) / float(reference_area) # 2800 = dx

    # Compute pairwise correlation for each interior particle
    for p in range(num_interior_particles):
        index = interior_indices[p]
        d = np.sqrt((x[index] - x)**2 + (y[index] - y)**2)
        d[index] = 2 * r_max   # Because sqrt(0)

        result, bins = np.histogram(d, bins=edges, normed=False)
        if normalize:
            result = result/number_density
        g[p, :] = result

    # Average g(r) for all interior particles and compute radii
    g_average = np.zeros(num_increments)
    for i in range(num_increments):
        radii[i] = (edges[i] + edges[i+1]) / 2.
        rOuter = edges[i + 1]
        rInner = edges[i]
        if return_count_only==False:
            g_average[i] = np.mean(g[:, i]) / (np.pi * (rOuter**2 - rInner**2))
        else: # return only cloud number count without weighting it with annulus radius
            g_average[i] = np.mean(g[:, i])

    return g_average, radii, interior_indices



    

def use_calc_rdf(data,field,tresh,r_max, normalize=True, dx=2.8, dr=1, mask=None, identify_clouds_kws = dict(), return_count_only =False, **kws):
    """
    Rechnet aufgrund der fünfzehnminütigen Niederschlagsdifferenz den Stündlichen Niederschlag hoch.
    Benötigt die Funktionen (Stephan):
        identify_clouds
        calc_rdf
        pair_correlation_2d
     Benötigt die Funktionen (Neu):   
        hourly_PREC_amount
    
    Parameters
    ----------
    data : Dataset womit das Ergebnis verknüpft werden soll
    inargs : Variable des Datasets mit aufsummierten Niederschlag
    field: Precipitation
    thresh : float
      Threshold for field
    r_max : int, optional
      Maximum search radius for RDF algorithm (in grid pts)
              ((Selber Wert wie bei calc_rdf))
    Returns
    -------
    rdf_set : Dataset mit:
                    RDF : (Normalized) Radial Distribution Function
                    radii : Distance
                    labels : labels
    """
    
    threshold=tresh
    #Liste mit Zeitschritt-index erstellen:
    time_list=field.time.values
    
    #Notwendige Listen anlegen:
    labels_list=[]
    rdf_list=[]
    radii_list=[]
    
    #fill field_list:
    for t in time_list:
        field_sel = field.sel(time=t).values
        labels=identify_clouds(field_sel,threshold,**identify_clouds_kws)[0]
        rdf, radii=calc_rdf(labels, field_sel, normalize=normalize, dx=dx, r_max=r_max, dr=dr, mask=mask,return_count_only=return_count_only, **kws)
        
        
        labels_list.append(labels)
        rdf_list.append(rdf)
        radii_list.append(radii)
    #DataSet RDF,radii,labels
    coords={'time':data.time.values,'Radius':radii,'rlat':data.rlat.values,'rlon':data.rlon.values}
    rdf_set=xr.Dataset({'RDF':(['time','Radius'],rdf_list),'radii':(['time','Radius'],radii_list),'labels':(['time','rlat','rlon'],labels_list)},coords=coords)
    rdf_set.RDF.attrs=dict(standard_name='(Normalized) RDF',long_name='(Normalized) Radial Distribution Function')
    rdf_set.Radius.attrs=dict(standard_name='Radius',long_name='Radius',units='km')
    rdf_set.time.attrs=dict(standard_name='time',long_name='time',bounds='time_bnds')
    rdf_set.rlat.attrs=dict(standard_name='grid_latitude',long_name='rotated latitude',units='degrees')
    rdf_set.rlon.attrs=dict(standard_name='grid_longitude',long_name='rotated longitude',units='degrees')
    return rdf_set


def main():
    from misc_functions import get_prec_data, plot_prec_fields, get_radar_mask
    selected_run_list =  ['reference', 'radar']
    date_list = ['20160527','20160528','20160529', '20160530','20160531', 
                '20160601','20160602', '20160603','20160604', '20160605','20160606', '20160607','20160608', '20160609']
    prec_list=[]
    for date in date_list:
        prec_list.append( get_prec_data(run_names = selected_run_list,date=date) )
        

        #radar_mask = get_radar_mask(date=date, use_time_dependent=False, date=date)
    prec_arrays = xr.concat(prec_list,'time')




# -----------------------------------------------------------------------------
# -------------------------- Main execution of Code ---------------------------
# -----------------------------------------------------------------------------    
if __name__=='__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s:%(message)s')
    
    print('tests')
    x  =np.arange(0,10,1)
    #fire.Fire(main)
