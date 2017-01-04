
import sys, os, glob
import logging
import numpy as np 

import var_def 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def read_tracks_parameters_to_ascii(ascii_in):
  """
  This routine reads the contents of an ascii file which tabulates the track parameters, and returns
  a list of "var_def.track()" objects, one per each row in the file. The list can be used later on
  for any manipulation (plotting, inserting into the database, etc). Note that we skip the first row
  as the header.

  @param ascii_in: the full path to the already-available ascii file that contains the entire (or part)
         of the tracks parameters. This file can be generated by first calling the function  
         write_tracks_parameters_to_ascii().
  @type ascii_out: string
  @return: list of instances of var_def.track() class objects, one object per each row (i.e. track).
  @rtype: list
  """
  if not os.path.exists(ascii_in):
    logging.error('read_tracks_parameters_to_ascii: {0} does not exist'.format(ascii_in))

  with open(ascii_in, 'r') as r: lines = r.readlines()
  header  = lines.pop(0)
  n_lines = len(header)
  list_tracks = []

  for i, line in enumerate(lines):
    row   = line.rstrip('\r\n').split(' ')
    M_ini = float(row[0])
    fov   = float(row[1])
    Z     = float(row[2])
    logD  = float(row[3])

    a_track = var_def.track(M_ini=M_ini, fov=fov, Z=Z, logD=logD)
    list_tracks.append(a_track)

  logging.info('insert_lib: read_tracks_parameters_to_ascii saved {0}'.format(ascii_in))
  print ' - insert_lib: read_tracks_parameters_to_ascii saved {0}'.format(ascii_in)

  return list_tracks

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
    logging.error('insert_lib: write_tracks_parameters_to_ascii: No track data stored. Call get_track_parameters() first')

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
  logging.info('insert_lib: write_tracks_parameters_to_ascii saved {0}'.format(ascii_out))
  print ' - insert_lib: write_tracks_parameters_to_ascii saved {0}'.format(ascii_out)

  return True

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
