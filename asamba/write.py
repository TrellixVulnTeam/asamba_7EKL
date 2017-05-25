from __future__ import print_function
from __future__ import unicode_literals

import sys, os, glob
import logging
import numpy as np 
import h5py

from asamba import var_lib, read, utils

import time

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logging.getLogger(__name__)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def write_sampling_to_h5(self_sampling, h5_out, include_periods=False):
  """
  This routine writes the learing_x and learning_y sets compiled from the sampler module as an 
  HDF5 file. This file is useful to export to other users, or save for a specific purpose. The format
  of the output file is the following: each row corresponds to one learning/training example. 

  - Then the first 6 columns correspond to initial mass ('M_ini'), exponential overshooting parameter ('fov'), 
    metallicity ('Z'), logarithm of the extra diffusive mixing ('logD'), age (measured as the core hydrogen
    mass fraction, 'Xc'), and rotation rate w.r.t. the critical break up rotation frequency ('eta').

  - The next K columns correspond to the K frequencies that are selected based on the K modes that the 
    star exhibits. These columns are labelled as 'f_1', 'f_2', ..., 'f_K'

  - Optionally, the periods can also be included, which is not a very novel inclusion (but good for 
    lazy people). If selected, another K columns will be included corresponding to periods of the modes
    1 to K. These columns are labelled as 'per_1', 'per_2', ..., 'per_K'

  Notes: 
  - The dataset which sits at the root group is named *learning_set*, which can be used to retrieve/read the data.
  - To read the file back, and recover the learning set, you can call the function read.read_from

  @param self_sampling: an instance of the sampler.sampling() class
  @type self_sampling: object
  @param h5_out: The output path to store the HDF5 file
  @type h5_out: str
  @param include_periods: flag to include the mode periods per each row, or not.
  @type include_periods: boolean
  @return: True if all went well, or False otherwise
  @rtype: boolean
  """
  ss = self_sampling
  if not ss.get('learning_done'):
    logger.warning('write_sampling_to_h5: The learning is not done yet! Skipping')
    return False

  # retrieve the necessary data
  x     = ss.get('learning_x')
  y     = ss.get('learning_y')
  names = ss.get('feature_names')
  flag  = ss.get('exclude_eta_column')
  if flag: names.extend(['eta'])

  # sizes of the data
  mx, n = x.shape
  if flag: n += 1
  my, K = y.shape
  try: 
    assert mx == my
  except:
    logger.error('write_sampling_to_h5: The X and Y matrixes have different number of rows!')
    return False

  f_names = ['f_{0}'.format(k) for k in range(K)]
  p_names = ['per_{0}'.format(k) for k in range(K)] if include_periods else []
  _names  = names + f_names + p_names
  types   = np.dtype( [(_name, 'f4') for _name in _names] )

  # load the data matrix with the correct columns
  arrs     = [x]
  if flag:  # eta column is not in x, so we put a column of zeros instead
    arrs.append(np.zeros((mx, 1)))
  arrs.append(y)
  if include_periods:
    arrs.append(1.0/y)
  data = np.concatenate(arrs, axis=1)

  # dump the data down now as a HDF5 file
  with h5py.File(h5_out, 'w') as h5:
    name = 'learning_set'
    dset = h5.create_dataset(name, data=data.astype(types), shape=data.shape, 
                             compression='gzip', compression_opts=9)

  logger.info('write_sampling_to_h5: saved {0}'.format(h5_out))

  return True

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def write_model_parameters_to_ascii(self_models, ascii_out):
  """
  Note: The old ascii_out file will be overwritten, if it already exists.
  """
  t0 = time.time()
  sm = self_models
  n_models = sm.get_n_models()
  if n_models == 0:
    logger.warning('write_model_parameters_to_ascii: calling models.find_list_filenames() first')
    sm.find_list_filenames()
  t1 = time.time()
  list_gyre_in = sm.get_list_filenames()
  n_files      = len(list_gyre_in)
  logger.info('write_model_parameters_to_ascii: Found "{0}" "{1}" files.'.format(
              n_files, sm.get_model_extension()))
  t2 = time.time()
  sm.sort_list_filenames()
  t3 = time.time()
  print('t1-t0: time to find files:', t1-t0)
  print('t2-t1: get list of filenames:', t2-t1)
  print('t3-t2: sort filenames:', t3-t2) 

  # get model attributes other than the six basic ones
  other_attrs   = var_lib.get_model_other_attrs()
  color_attrs   = var_lib.get_model_color_attrs()
  rec_attrs     = []
  for attr in other_attrs:
    if attr in color_attrs and '_' in attr:
      attr      = attr.replace('_', '-')
    rec_attrs.append(attr)
  set_attrs     = set(rec_attrs)

  # open the file handle
  if os.path.exists(ascii_out): os.unlink(ascii_out)
  try:
    handle    = open(ascii_out, 'a')
  except:
    logger.error('write_model_parameters_to_ascii: failed to open: "{0}"'.format(ascii_out))
    sys.exit(1)

  # collect the header for all columns
  header      = '{0:>6s} {1:>5s} {2:>5s} {3:>5s} {4:>6s} {5:>5s} '.format(
                 'M_ini', 'fov', 'Z', 'logD', 'Xc', 'num')
  header      += ' '.join([ '{0:>12s}'.format(attr[:12]) for attr in rec_attrs ]) + '\n'
  # write the header
  handle.write(header)

  # iterate over the list of input GYRE files, and fetch the corresponding info from the history file  
  last_histname = ''
  for i, filename in enumerate(list_gyre_in):

    # find the corresponding history file for this model
    histname  = var_lib.gen_histname_from_gyre_in(filename)
    # avoid reading the hist file again if the model is along the same track as that of the 
    # previous iteration
    if histname == last_histname:
      pass
    else:
      if not os.path.exists(histname):
        logger.error('write_model_parameters_to_ascii: missing the corresponding hist file {0}'.format(histname))
        sys.exit(1)
      hdr, hist = read.read_mesa_ascii(histname)
      last_histname = histname

    tup_gyre_in_par = var_lib.get_model_parameters_from_gyre_in_filename(filename)

    M_ini     = tup_gyre_in_par[0]
    fov       = tup_gyre_in_par[1]
    Z         = tup_gyre_in_par[2]
    logD      = tup_gyre_in_par[3]
    evol_state= tup_gyre_in_par[4]
    Xc        = tup_gyre_in_par[5]
    model_number = tup_gyre_in_par[6]

    # get the corresponding row for this model from the hist recarray
    ind_row   = model_number - 1
    if model_number == hist['model_number'][ind_row]:
      pass
    else:
      logger.error('write_model_parameters_to_ascii: messed up model_number!')
      ind_row = np.where(hist['model_number'] == model_number)[0][0]
    row       = hist[ind_row]

    # manually, construct the first 6 columns of the output file
    line      = '{0:>06.3f} {1:>05.3f} {2:>05.3f} {3:>05.2f} {4:>06.4f} {5:>05d} '.format(
                 M_ini, fov, Z, logD, Xc, model_number)
    line      += ' '.join(['{0:>12.5e}'.format(row[attr]) for attr in rec_attrs]) + ' \n'

    # append to the ascii file, and to the output list
    handle.write(line)

  logger.info('write_model_parameters_to_ascii: saved "{0}"'.format(ascii_out))
  print(' - asamba.write.write_model_parameters_to_ascii: saved "{0}"'.format(ascii_out))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def write_model_parameters_to_ascii_obsolete(self_models, ascii_out):
  """
  Note: The old ascii_out file will be overwritten, if it already exists.
  """
  sm = self_models
  n_models = sm.get_n_models()
  if n_models == 0:
    logger.error('write_model_parameters_to_ascii_obsolete: the passed "models" object has no models inside')
    sys.exit(1)
  list_models = sm.get_list_models()

  # open the file handle
  if os.path.exists(ascii_out): os.unlink(ascii_out)
  try:
    handle    = open(ascii_out, 'a')
  except:
    logger.error('write_model_parameters_to_ascii_obsolete: failed to open: "{0}"'.format(ascii_out))
    sys.exit(1)

  # filter the attributes
  first_model = list_models[0]
  avail_attrs = dir(first_model)
  exclude     = set(['__init__', '__doc__', '__module__', 'filename', 'track'])
  avail_attrs = [attr for attr in avail_attrs if attr not in exclude]
  avail_attrs = [attr for attr in avail_attrs if 'set' not in attr and 'get' not in attr]
  
  key_attrs   = ['M_ini', 'fov', 'Z', 'logD', 'Xc', 'model_number']
  # key_fmt     = [float, float, float, float, str, float, int]
  other_attrs = [attr for attr in avail_attrs if attr not in key_attrs]

  # collect the header for all columns
  header      = '{0:>6s} {1:>5s} {2:>5s} {3:>5s} {4:>6s} {5:>5s} '.format(
                 'M_ini', 'fov', 'Z', 'logD', 'Xc', 'num')
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
    line      = '{0:>06.3f} {1:>05.3f} {2:>05.3f} {3:>05.2f} {4:>06.4f} {5:>05d} '.format(
                 model.M_ini, model.fov, model.Z, model.logD, model.Xc, model.model_number)
    
    # iterate over the rest of the attributes, and convert them to string
    for k, attr in enumerate(other_attrs): 
      line += '{0:>12.5e} '.format(getattr(model, attr)[0])
    line += '\n'

    # append to the ascii file, and to the output list
    handle.write(line)
    lines.append(line)

  logger.info('write_model_parameters_to_ascii_obsolete: saved "{0}"'.format(ascii_out))
  print(' - asamba.write.write_model_parameters_to_ascii_obsolete: saved "{0}"'.format(ascii_out))

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
  print(' - write: write_tracks_parameters_to_ascii saved {0}'.format(ascii_out))

  return True

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def Xc_tags_to_ascii(dic_tags, ascii_out='data/tags/Xc-tags.txt'):
  """
  The db_lib.get_dic_tag_Xc() function returns a dictionary that helps tagging Xc (or equivalently)
  the MESA models, based on their attributes (M_ini, fov, Z, logD, Xc). The retrieval of the 
  information needed to construct the dictionary takes at least 3 minutes (and longer if connecting
  the network, if the data has to be transferred from a cloud/HTTP host). It is then easier
  to write this data to the disk as an ascii file, and read it each time.

  Note: 
  - The Xc_tags_to_h5() method has an indentical operation, but stores the data as an HDF5 file, 
    gaining up to ~10 times better compression (smaller sized file). Consider that, instead.
  - To read the data, you can call the function read.Xc_tags_from_ascii().

  @param dic_tags: a dictionary that contains the model attributes as keys and Xc tags as values, 
       returned by calling db_lib.get_dic_tag_Xc().
  @type dic_tags: dict
  @param ascii_out: full path to store the tags as an ascii file
  @type ascii_out: str
  @return: True if all succeeds, and False otherwise
  @rtype: boolean
  """
  keys  = sorted(list(dic_tags.keys()))
  tags  = [dic_tags[key] for key in keys]
  rows  = sorted([key + (tag, ) for key, tag in list(zip(keys, tags))])
  lines = ['M_ini,fov,Z,logD,Xc,tag\n']
  lines += ['{0:06.3f},{1:05.3f},{2:05.3f},{3:05.2f},{4:06.4f},{5:d}\n'.format(*row) for row in rows]
  try:
    with open(ascii_out, 'w') as w: w.writelines(lines)
    logger.info('Xc_tags_to_ascii: saved the file {0}'.format(ascii_out))
    return True
  except:
    logger.warning('Xc_tags_to_ascii: Failed. Check ascii_out path first!')
    return False

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def Xc_tags_to_h5(dic_tags, h5_out='data/tags/Xc-tags.h5'):
  """
  This function is identical to Xc_tags_to_ascii(), but dumps the output as an HDF5 file. Refer to
  Xc_tags_to_ascii() for more details.
  The compression gain here compared to saving an ASCII file is roughly ~10 times! So, we highly 
  recommend using this, instead of Xc_tags_to_ascii(). To read the data, you can call the function 
  read.Xc_tags_from_h5().

  In fact, the h5py Python modules has a too friendly relashionship with numpy, and the data are 
  converted silently to numpy array, even if not explicitly asked through the create_dataset() arguments.
  For this reason, the attribute formats become screwed again, and I strongly discourage using this 
  routine. One will be better off with Xc_tags_to_ascii() -- at the cost of speed and file volume.
  """
  keys  = sorted(list(dic_tags.keys()))
  tags  = [dic_tags[key] for key in keys]
  rows  = [key + (tag, ) for key, tag in list(zip(keys, tags))]
  n, m  = len(rows), 6
  try:
    with h5py.File(h5_out, 'w') as h5:
      dset = h5.create_dataset('Xc_tags', data=rows, shape=(n, m), 
                               compression='gzip', compression_opts=9)
    logger.info('Xc_tags_to_h5: saved the file {0}'.format(ascii_out))
    return True
  except:
    logger.warning('Xc_tags_to_h5: Failed. Check ascii_out path first!')
    return False
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
