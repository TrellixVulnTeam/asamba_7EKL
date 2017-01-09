
import sys, os, glob
import logging
import numpy as np 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logging.getLogger(__name__)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def write_model_parameters_to_ascii(self_models, ascii_out):
  """

  """
  sm = self_models
  n_models = sm.get_n_models()
  if n_models == 0:
    logger.error('write_model_parameters_to_ascii: the passed "models" object has no models inside')
    sys.exit(1)

  avail_attrs = dir(sm)
  exclude     = set(['__init__', '__doc__', '__module__'])
  avail_attrs = [attr for attr in avail_attrs if attr not in exclude]
  avail_attrs = [attr for attr in avail_attrs if 'set' not in attr or 'get' not in attr]
  print avail_attrs

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def write_tracks_parameters_to_ascii(self_tracks, ascii_out):
  """
  Store the four parameters of the MESA tracks (mass, overshoot, metallicity and extra mixing) as
  an ascii file on the disk. To do so, the var_def.get_track_parameters() method must have already
  been applied on the var_def.tracks() class object. 
  The format of the stored file is the following: the parameters in each row correspond to one track.
  There will be four columns, separated by a single space, and they correspond to the initial mass
  (M_ini), core overshooting parameter (fov), metallicity (Z), and extra diffusive mixing (logD),
  respectively.

  @param self_tracks: an instance of the var_def.tracks()
  @type self_tracks: class object
  @param ascii_out: full path to store the track parameters.
  @type ascii_out: string
  """
  if self_tracks.n_tracks == 0:
    logger.error('write_tracks_parameters_to_ascii: No track data stored. Call get_track_parameters() first')
    sys.exit(1)

  # add a header
  lines       = ['{0:<6s} {1:<5s} {2:<5s} {3:<5s} \n'.format('M_ini', 'fov', 'Z', 'logD')]

  list_tracks = self_tracks.list_tracks
  for i, obj in enumerate(list_tracks):
    str_M_ini = '{0:06.3f}'.format(obj.M_ini)
    str_fov   = '{0:05.3f}'.format(obj.fov)
    str_Z     = '{0:05.3f}'.format(obj.Z)
    str_logD  = '{0:05.2f}'.format(obj.logD)
    line      = '{0} {1} {2} {3} \n'.format(str_M_ini, str_fov, str_Z, str_logD)
    lines.append(line)

  with open(ascii_out, 'w') as w: w.writelines(lines)
  logger.info('write_tracks_parameters_to_ascii saved {0}'.format(ascii_out))
  print ' - write: write_tracks_parameters_to_ascii saved {0}'.format(ascii_out)

  return True

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
