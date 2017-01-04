
import sys, os, glob
import logging
import numpy as np 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class tracks:
  """
  Class object that agglemerates multiple instances of the "track" object
  """
  def __init__(self):
    self.n_tracks = 0
    self.list_dic_tracks = []

  def set_n_tracks(self, n_tracks):
    self.n_tracks = n_tracks

  def set_list_dic_tracks(self, list_dic_tracks):
    self.list_dic_tracks = list_dic_tracks

  def get_n_tracks(self):
    return self.n_tracks

  def get_list_dic_tracks(self):
    return self.list_dic_tracks

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class track:
  """
  Class object that stores the data for MESA tracks
  """
  def __init__(self, M_ini, Z, fov, logD):
    """
    Constructor that stores the mass, metalicity, overshoot and extra diffusive mixing per each 
    track
    """
    self.M_ini = M_ini
    self.Z = Z
    self.fov = fov
    self.logD = logD

    self.hist_string = ''

  def set_M_ini(self, M_ini):
    self.M_ini = M_ini

  def set_Z(self, Z):
    self.Z = Z

  def set_fov(self, fov):
    self.fov = fov

  def set_logD(self, logD):
    self.logD = logD

  def set_hist_string(self, hist_string):
    self.hist_string = hist_string

  def get_M_ini(self):
    return self.M_ini

  def get_Z(self):
    return self.Z

  def get_fov(self):
    return self.fov 

  def get_logD(self):
    return self.logD 

  def get_hist_string(self):
    return self.hist_string

  def get_dic_track_parameters(self):
    return {'M_ini':self.M_ini,
            'Z':self.Z,
            'fov':self.fov,
            'logD':self.logD}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
