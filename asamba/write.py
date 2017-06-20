from __future__ import print_function
from __future__ import unicode_literals

import sys, os, glob
import logging
import numpy as np 
import h5py

from asamba import var_lib, db_lib, read, utils

import time

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logger = logging.getLogger(__name__)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def write_rotation_frequencies_to_ascii(dbname, h5_files, ascii_out):
  """
  This routine receives the names of HDF5 GYRE files as input, and collects the following attributes
  from the file header, if available (else it quits). Then, the whole information is collected for
  output as an ASCII file. This is useful to COPY the contents of the created file into the corresponding
  table rotation_frequencies in the database.
  To read the created file, you may use read.read_rotation_frequencies_from_ascii().

  @param dbname: the name of the database to connect to (and retrieve the look up dictionaries)
  @type dbname: str
  @param h5_files: the full path to the whole h5 files to add to the output file
  @type h5_files: list of strings
  @param ascii_out: the full path to the location of the created file
  @type ascii_out: str
  @return: True if successful, and False, otherwise.
  @rtype: bool
  """
  n_h5 = len(h5_files)
  if n_h5 == 0:
    logger.error('write_rotation_frequencies_to_ascii: The input file is empty')
    sys.exit(1)

  # make sure the attributes that we need are available
  first = h5_files[0]
  with h5py.File(first, 'r') as h5:
    keys  = h5.attrs.keys()
  if 'freq_rot' not in keys or 'freq_crit' not in keys:
    logger.warning('write_rotation_frequencies_to_ascii: the extra rotation attributes unavailable')
    return False

  dic_tracks_id = db_lib.get_dic_look_up_track_id(dbname)
  dic_models = db_lib.get_dic_look_up_models_id(dbname)
  dic_rot    = db_lib.get_dic_look_up_rotation_rates_id(dbname)

  id_nonrot  = dic_rot[ (0.0, ) ]

  lines      = ['id_model,id_rot,freq_crit,freq_rot\n']
  for k, f in enumerate(h5_files):
    # Retrieve the id_model and id_rot from the filename
    params       = var_lib.get_model_parameters_from_gyre_out_filename(filename=h5_file)
    M_ini        = params[0]
    fov          = params[1]
    Z            = params[2]
    logD         = params[3]
    Xc           = params[4]
    model_number = params[5]
    eta          = params[6]

    tup_track    = (M_ini, fov, Z, logD)
    id_track     = dic_tracks_id[tup_track]

    tup_model    = (id_track, model_number)
    id_model     = dic_models_id[tup_model]

    tup_eta      = (eta, )
    id_rot       = dic_rot_rates[tup_eta]

    # Retrieve the rotation and break up frequencies from the file attributes
    if id_rot == id_nonrot:
      freq_rot   = 0.0
      freq_crit  = 0.0           # a nonrotating model has a critical break up, but it is not stored
    else:
      with h5py.File(f, 'r') as h5: 
        freq_rot = h5.attrs['freq_rot']
        freq_crit= h5.attrs['freq_crit']

    line         = '{0},{1},{2:.6e},{3:.6e}\n'.format(id_model, id_rot, freq_crit, freq_rot)
    lines.append(line)

    # progress bar
    if k % 1000 == 0:
      sys.stdout.write('\r')
      sys.stdout.write('progress = {0:.2f} % '.format(100*float(k)/n_h5)) 
      sys.stdout.flush()


  with open(ascii_out, 'w') as w: w.writelines(lines)

  logger.info('write_rotation_frequencies_to_ascii: done')

  return True

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def write_sampling_to_h5(self_sampling, h5_out, include_periods=False):
  """
  This routine writes the sampling datasets compiled from the sampler module as an 
  HDF5 file. This file is useful to export to other users, or save for a specific purpose. The format
  of the output file is the following: each row corresponds to one learning/training example, and the data is 
  structured as several HDF5 datasets. Using this HDF5 format, the sampling data are very portable,
  independent of the platform used.

  The available datasets are the following:
  - learning_ids_models: has only one column and stores the models.id for each record

  - learning_ids_rot: has only one column and stores the rotation_rate.id for each record

  - learning_x: has 6 columns corresponding to initial mass ('M_ini'), exponential overshooting parameter ('fov'), 
    metallicity ('Z'), logarithm of the extra diffusive mixing ('logD'), age (measured as the core hydrogen
    mass fraction, 'Xc'), and rotation rate w.r.t. the critical break up rotation frequency ('eta').

  - learning_y: has K columns corresponding to the K frequencies that are selected based on the K modes that the 
    star exhibits. These columns are labelled as 'f_0', 'f_1', ..., 'f_{K-1}'

  - learning_radial_orders: has K columns corresponding to K frequencies, and takes care of the bookkeeping of the
    radial order (n_pg) for each mode per each record. These columns are labelled as 'n_0', ..., 'n_{K-1}'
  
  - learning_mode_types: has K columns corresponding to K frequencies, and carries the mode type label per each mode
    per each record. These columns are labelled as 't_0', ..., 't_{K-1}'

  - Optionally, the periods can also be included, which is not a very novel inclusion (but good for 
    lazy people). If selected, another K columns will be included corresponding to periods of the modes
    1 to K. These columns are labelled as 'per_0', 'per_1', ..., 'per_{K-1}'

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
  idmod = ss.get('learning_ids_models')
  idrot = ss.get('learning_ids_rot')

  x     = ss.get('learning_x')
  mx, n = x.shape
  y     = ss.get('learning_y')
  my, K = y.shape
  names = ss.get('feature_names')
  flag  = ss.get('exclude_eta_column')
  if flag: 
    x   = np.concatenate( [ x, np.zeros((mx, 1)) ], axis=1)
    n   += 1
    names.extend(['eta'])

  # consistent sizes of the data
  try: 
    assert mx == my
  except:
    logger.error('write_sampling_to_h5: The X and Y matrixes have different number of rows!')
    return False

  # other valuable learning data
  n_pg  = ss.get('learning_radial_orders')
  lmt   = ss.get('learning_mode_types')

  log_Teff = ss.get('learning_log_Teff')
  log_g = ss.get('learning_log_g')

  f_names = ['f_{0}'.format(k) for k in range(K)]   # for learning_y
  p_names = ['per_{0}'.format(k) for k in range(K)] if include_periods else []
  n_names = ['n_{0}'.format(k) for k in range(K)]   # for learning_radial_orders
  t_names = ['t_{0}'.format(k) for k in range(K)]   # for learning_mode_types

  # dump the data down now as a HDF5 file as different datasets
  with h5py.File(h5_out, 'w') as h5:
    # learning_ids_models
    dset_i = h5.create_dataset('learning_ids_models', data=idmod, shape=idmod.shape, dtype=int,
              compression='gzip', compression_opts=9)
    dset_i.attrs['num_rows']    = idmod.shape[0]
    dset_i.attrs['num_columns'] = 1
    dset_i.attrs['column_names']= np.array(['id_model'], 'S6')

    # learning_ids_rot
    dset_r = h5.create_dataset('learning_ids_rot', data=idrot, shape=idrot.shape, dtype=np.int16,
              compression='gzip', compression_opts=9)
    dset_r.attrs['num_rows']    = idrot.shape[0]
    dset_r.attrs['num_columns'] = 1
    dset_r.attrs['column_names']= np.array(['id_rot'], 'S6')

    # learning_radial_orders
    dset_n = h5.create_dataset('learning_radial_orders', data=n_pg, shape=n_pg.shape, dtype=np.int32,
              compression='gzip', compression_opts=9)
    dset_n.attrs['num_rows']    = n_pg.shape[0]
    dset_n.attrs['num_columns'] = n_pg.shape[1]
    dset_n.attrs['column_names']= np.array(n_names, 'S6')

    # learning_mode_types
    dset_t = h5.create_dataset('learning_mode_types', data=lmt, shape=lmt.shape, dtype=np.int16,
              compression='gzip', compression_opts=9)
    dset_t.attrs['num_rows']    = lmt.shape[0]
    dset_t.attrs['num_columns'] = lmt.shape[1]
    dset_t.attrs['column_names']= np.array(t_names, 'S6')

    # Learning set for features
    dset_x = h5.create_dataset('learning_x', data=x, shape=x.shape, dtype='f4', 
              compression='gzip', compression_opts=9)
    dset_x.attrs['num_rows']     = mx 
    dset_x.attrs['num_columns']  = len(names)
    dset_x.attrs['column_names'] = np.array(names, 'S6')

    # Learning set for frequencies
    dset_y = h5.create_dataset('learning_y', data=y, shape=y.shape, dtype='f4', 
              compression='gzip', compression_opts=9)
    dset_y.attrs['num_rows']     = my
    dset_y.attrs['num_columns']  = K
    dset_y.attrs['column_names'] = np.array(f_names, 'S6')

    # Learning set for periods
    if include_periods:
      dset_p = h5.create_dataset('learning_periods', data=1.0/y, shape=y.shape, dtype='f4', 
                compression='gzip', compression_opts=9)
      dset_p.attrs['num_rows']     = my
      dset_p.attrs['num_columns']  = K
      dset_p.attrs['column_names'] = np.array(p_names, 'S6')

    # Model effective temperatures
    dset_Tf= h5.create_dataset('learning_log_Teff', data=log_Teff, shape=log_Teff.shape, dtype='f4', 
              compression='gzip', compression_opts=9)
    dset_Tf.attrs['num_rows']    = mx 
    dset_Tf.attrs['num_columns'] = 1
    dset_Tf.attrs['column_names']= np.array(['log_Teff'], 'S8')

    # Model surface gravity
    dset_g = h5.create_dataset('learning_log_g', data=log_g, shape=log_g.shape, dtype='f4', 
              compression='gzip', compression_opts=9)
    dset_g.attrs['num_rows']     = mx 
    dset_g.attrs['num_columns']  = 1
    dset_g.attrs['column_names'] = np.array(['log_g'], 'S6')

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
