
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
  list_models = sm.get_list_models()

  # open the file handle
  if os.path.exists(ascii_out): os.unlink(ascii_out)
  try:
    handle    = open(ascii_out, 'a')
  except:
    logger.error('write_model_parameters_to_ascii: failed to open: "{0}"'.format(ascii_out))
    sys.exit(1)

  # filter the attributes
  first_model = list_models[0]
  avail_attrs = dir(first_model)
  exclude     = set(['__init__', '__doc__', '__module__', 'filename', 'track'])
  avail_attrs = [attr for attr in avail_attrs if attr not in exclude]
  avail_attrs = [attr for attr in avail_attrs if 'set' not in attr and 'get' not in attr]
  
  key_attrs   = ['M_ini', 'fov', 'Z', 'logD', 'evol', 'Xc', 'model_number']
  # key_fmt     = [float, float, float, float, str, float, int]
  other_attrs = [attr for attr in avail_attrs if attr not in key_attrs]

  # collect the header for all columns
  header      = '{0:>6s} {1:>5s} {2:>5s} {3:>5s} {4:>4s} {5:>6s} {6:>5s}'.format(
                 'M_ini', 'fov', 'Z', 'logD', 'evol', 'Xc', 'num')
  for attr in other_attrs:
    header    += '{0:>12s} '.format(attr[:12])
  header      += '\n'

  # write the header
  handle.write(header)

  # collect the line info as lines
  lines       = [header]

  # iterate over models, and collect data into lines
  for i, model in enumerate(list_models):
    # first, the key attributes
    line      = '{0:>06.3f} {1:>05.3f} {2:>05.3f} {3:>05.2f} {4:>4s} {5:>06.4f} {6:>05d} '.format(
                 model.M_ini, model.fov, model.Z, model.logD, model.evol, model.Xc, model.model_number)
    
    # iterate over the rest of the attributes, and convert them to string
    for k, attr in other_attrs: line += '{0:>12.6e3} '.format(getattr(model, attr))
    line += '\n'

    # append to the ascii file, and to the output list
    handle.write(line)
    lines.append(line)

  logger.info('write_model_parameters_to_ascii: saved "{0}"'.format(ascii_out))
  print ' - grid.write.write_model_parameters_to_ascii: saved "{0}"'.format(ascii_out)

  return lines

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
