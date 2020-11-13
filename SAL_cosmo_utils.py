"""compute the SAL-score (Structure, Amplitude, Location)

The SAL score is an object-based score for
Quantitative Precipitation Forecasts (QPF)

The implementation here follows

Wernli_et_al_2008_SAL_score_MWR_136_4470-4487

with the help of some ``scipy``-functions for object separation.
"""

#
# Author: Heiner Lange
# heiner.lange@physik.uni-muenchen.de
#
#
# modified by Mirjam Hirt, June 2018
# The computation of SAL threshold is now according to a percentile instead
# of maximum. Also, for obs and mod different threshold are considered!
#

# $Id$

import numpy as np
from scipy import ndimage
from numpy.linalg import norm



def threshold_to_zero(field):
  """thresholds a numpy-field to zero if it contains negative values

  input: numpy-array

  output: numpy-array, all negative values set to 0.
  """

  field = np.where(field < 0., 0., field)

  return field



def compute_SAL(field_obs, field_fc, user_threshold=None):
  """SAL-score (Structure, Amplitude, Location)

  nach Wernli_et_al_2008_SAL_score_MWR_136_4470-4487
  If no user_threshold is given:
  --> For each 2d field, the 95th percentile is computed for both obs and
  model for values, where precipitation>0.1. This is divided by 15 to
  yield a

  Parameters
  ----------
  field_obs : numpy.ndarray
    2D observation (or reference) field
  field_fc : numpy.ndarray
    2D forecast field (same shape as `field_obs`)
  user_threshold : float, optional
    If set, use this threshold for object separation.
    If not set, use the default method of Wernli is used: ``threshold = field_obs.max() * 1/15``

  Returns
  -------
  S : float
    Structure score
  A : float
    Amplitude score
  L : float
    Location score
  L1, L2 : float
    Location score components `L1` and `L2`
  NObjects_obs : float
    Number of thresholded objects in observation
  NObjects_mod : float
    Number of objects in model (forecast `fc`)
  R_max : float
    Maximum Rain-value of observations `field_obs`
  user_treshold : float
    Threshold that was actually used

  Notes
  -----
  Geographical distances are internally treated in gridpoint-units
  (therefore, a horizontally regular grid is assumed).

  Examples
  --------
  >>> from cosmo_utils.pywgrib import getfobj
  >>> from cosmo_utils.scores.SAL import compute_SAL

  Read a nature run composite reflectivity field and compare it
  with an ensemble member. The threshold is costum set to 10 dBZ:

  >>> refl_nat = getfobj("lff20080730140000.nat","REFL_MAX")
  >>> refl_mem = getfobj("lff20080730140000.001","REFL_MAX")
  >>> compute_SAL(refl_nat.data, refl_mem.data, user_threshold=10.)
  (0.00608, -0.235671, 0.25080, 0.13973, 0.11106, 8, 9, 55.98, 10.0)

  Of course, it is more convenient to broadcast the score-results to variables
  for later use:

  >>> S, A, L, L1, L2, NObjects_obs, NObjects_mod, R_max, threshold = compute_SAL(refl_nat.data, refl_mem.data, user_threshold=10.)

  """

  # check: felder gleichfoermig:
  if (field_obs.shape != field_fc.shape):
    raise Exception("field_obs and field_fc need to have the same shape" +\
    " (this function also assumes that they cover the same domain!)")

  if (user_threshold is not None) and (type(user_threshold) != float):
    raise Exception("user_threshold must be a float")




  # regenfelder
  #
  # -> genannt "regen" -> kann aber auch ein anderes feld sein,
  # dessen grundwert 0 ist!
  #
  # umbenennung auf Wernli-Konvention

  R_mod = field_fc # "mod" is for "model"
  R_obs = field_obs # "obs" is for "observation"




  # checken, ob auch negative werte vorhanden sind.
  # falls ja, abbrechen
  #
  #

  if np.any(R_mod < 0.):

    raise Exception("negative values in forecast." +\
    "use different field or apply SAL.threshold_to_zero(field)")

  if np.any(R_obs < 0.):

    raise Exception("negative values in forecast." +\
    "use different field or apply SAL.threshold_to_zero(field)")



  # regenfelder maskieren

  f_threshold = 1/15. # relativer threshold laut Wernli

  R_max = np.nanpercentile(np.where(R_obs>0.1, R_obs, np.nan), 95)
  threshold_obs = R_max * f_threshold
  R_max = np.nanpercentile(np.where(R_mod>0.1, R_mod,np.nan), 95)
  threshold_mod = R_max * f_threshold

  if (user_threshold is not None):

    threshold_obs = user_threshold
    threshold_mod = user_threshold

  R_mod_thr = np.where(R_mod > threshold_mod, R_mod, 0.) # maskiert
  R_obs_thr = np.where(R_obs > threshold_obs, R_obs, 0.) # maskiert


  # amplitude (formel (2))

  D_R_mod = np.mean(R_mod)
  D_R_obs = np.mean(R_obs)

  A = (D_R_mod - D_R_obs) / (0.5 * (D_R_mod + D_R_obs))


  # location component L1 (formel (4))

  # x_R_mod: center of mass of fc [GridPoint-units]
  x_R_mod = np.array(ndimage.measurements.center_of_mass(R_mod))
  # x_R_obs: center of mass of obs [GP]
  x_R_obs = np.array(ndimage.measurements.center_of_mass(R_obs))

  # hinweis: x_R_mod/obs sind bei 2D-Felder Vektoren, bei 1D Feldern Skalare
  # die Funktion numpy.linalg.norm kommt mit beiden inputs gleichermassen zurecht

  d_diagonal = norm(field_obs.shape) # domain diagonal [GP]

  L1 = norm(x_R_mod - x_R_obs) / d_diagonal




  # location component L2 (formel (5)) und structure component S

  # objekterkennung via ndimage.measurements.label:
  #
  # labels_mod = array, in welchem verbundene objektflaechen durch
  # integer-flaechen ersetzt werden -> Objekt1 besteht aus 1-en,
  #                  Objekt2 aus 2-en etc.
  #                                    der rest des arrays ist 0
  #  NObjects_mod = anzahl der verbundenen objekte
  #
  # (verbindungsmatrix standard: [[0,1,0], [1,1,1], [0,1,0]])
  #
  #
  labels_mod, NObjects_mod = ndimage.measurements.label(R_mod_thr)
  labels_obs, NObjects_obs = ndimage.measurements.label(R_obs_thr)


  # listen fuer zwei durchlaufe der objekt-analyse: 0=mod, 1=obs

  labels_list = [labels_mod, labels_obs]
  NObjects_list = [NObjects_mod, NObjects_obs]

  R_list = [R_mod, R_obs]
  x_R_list = [x_R_mod, x_R_obs]

  r_list = [0,0] # fuer L2
  V_list = [0,0] # fuer S


  # zwei durchlaufe der objekt-analyse: iList=0=mod, iList=1=obs
  for iList, NObjects in enumerate(NObjects_list):

    # liste der regensummen der einzel-objekte
    R_n_list = np.zeros(NObjects)

    # liste der "scaled volumes" der einzel-objekte
    V_n_list = np.zeros(NObjects)

    # liste der schwerpunkte der einzel-objekte
    distance_n_list = np.zeros(NObjects)



    # objekte des feldes durchzaehlen
    for n in range(NObjects):

      label = n + 1 # labelzaehlung beginnt bei 1


      # array mit voller groesse. inhalt: objekt nummer "label", ansonsten 0
      object_n = np.where(labels_list[iList] == label, R_list[iList], 0)

      # Regensumme des objekts (formel vor formel (5))
      R_n_list[n] = object_n.sum()

      # V_n fuer S (formel (7)
      V_n_list[n] = R_n_list[n] / object_n.max()

      # schwerpunkt des objekts [GP]
      x_n = np.array(ndimage.measurements.center_of_mass(object_n))

      distance_n_list[n] = norm(x_R_list[iList] - x_n)





    # umwandlung der listen in arrays
    R_n_list = np.array(R_n_list)
    distance_n_list = np.array(distance_n_list)

    # formel (5):

    r_list[iList] = (R_n_list*distance_n_list).sum() / R_n_list.sum()


    # formel (8):

    V_list[iList] = (R_n_list*V_n_list).sum() / R_n_list.sum()



  # formel (6):

  L2 = 2.*norm(r_list[0] - r_list[1]) / d_diagonal

  L = L1 + L2



  # structure component S: formel (9)

  S = (V_list[0] - V_list[1]) / (0.5 * (V_list[0] + V_list[1])) # 0: mod, 1:obs


  #print "R_max = ", R_max
  #print "threshold = ", threshold # threshold that was actually applied for S,L
  #print "L1 = ", L1
  #print "L2 = ", L2
  #print "NObjects_obs = ", NObjects_obs
  #print "NObjects_mod = ", NObjects_mod


  return S, A, L, L1, L2, NObjects_obs, NObjects_mod, R_max, threshold_obs, threshold_mod