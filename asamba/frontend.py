#! /usr/bin/python

"""
This module is the user's frontend to open a graphical user interface, GUI, and interact with the 
underlying grid database and a (limited) suite of functionalities in the backend. There are millions 
of ways one can design the GUI, provide different functionalities, and/or allow the user to interact 
with the database. What is provided here is one way out of the millions of possibilities, and is a 
basic attempt to start with the GUI. Indeed, this GUI is pretty extensible, and flexible to grow and
accommodate additional tools for analysis.
"""

# To Do
#     # self.tab_seism.columnconfigure(0, weight=1)
#     # self.tab_seism.rowconfigure(0, weight=1)

from __future__ import division
from __future__ import unicode_literals

from future import standard_library
standard_library.install_aliases()
from builtins import next
from builtins import range
from builtins import object
from past.utils import old_div

import sys
from itertools import cycle
import logging

import numpy as np 

from tkinter import *
from tkinter import ttk
import tkinter.filedialog 
import tkinter.messagebox

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from asamba import backend as bk

####################################################################################
logger = logging.getLogger(__name__)

####################################################################################
class GUI(object):

  def __init__(self, root):
    """
    Get an instance of the frontend GUI.

    @param root: An instance of the tk.Tk() object
    @type root: object
    """
    self.root         = root     # The only input argument
    self.notebook     = 0

    # Global variables
    self.dir_data     = 'data/'
    self.dir_bitmaps  = self.dir_data + 'bitmaps/'

    # All Tabs
    self.tab_seism     = 0
    self.tab_query      = 0
    self.tab_sampling   = 0
    self.tab_ann        = 0
    self.tab_interp     = 0

    # # All useful frames and insets
    # self.frame_conn     = 0
    # self.frame_sample   = 0
    # self.frame_MAP      = 0
    # self.frame_stat_bar = 0

    # self.inputs_left    = 0
    # self.inputs_right   = 0

    # # Connection constant values
    self.connection   = IntVar()
    self.conn_loc     = 1   # Developer, accessing the 'asamba' database
    self.conn_loc_str = 'grid'
    self.conn_ivs     = 2   # user from the Institute of Astronomy, KUL
    self.conn_ivs_str = 'IvS'
    self.conn_https   = 3   # 'https://'
    self.conn_https_str = 'https://'
    self.dbname       = ''  # The final user's choice
    self.conn_status  = False

    self.conn_lbl_str = StringVar()
    self.bitmap_OK    = self.dir_bitmaps + 'OK.png'
    # self.handle_OK    = PhotoImage(self.bitmap_OK)
    self.bitmap_NOK   = self.dir_bitmaps + 'NOK.png'
    # self.handle_NOK   = PhotoImage(self.bitmap_NOK)

    # # Input frequency list
    # self.input_freq_file = StringVar()
    # self.input_freq_done = BooleanVar()

    # # Input observational data
    # self.use_obs_log_Teff = BooleanVar()
    # self.obs_log_Teff     = DoubleVar()
    # self.obs_log_Teff_err = DoubleVar()

    # self.use_obs_log_g    = BooleanVar()
    # self.obs_log_g        = DoubleVar()
    # self.obs_log_g_err    = DoubleVar()

    # # Sampling 
    # self.which_sampling_function = BooleanVar()
    # self.sampling_constrained    = True 
    # self.sampling_randomly       = False
    # self.sampling_shuffle        = BooleanVar()        

    # # Status bar messages
    self.status_bar_message = StringVar()

    # Trigger the FrontWindow now ...
    self.FrontWindow()      

  def FrontWindow(self):
    """ The front GUI window """

    ##########################################
    # The root and the notebook
    ##########################################
    root       = self.root
    root.title('ASAMBA User Interface')
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    self.notebook   = ttk.Notebook(root)
    style = ttk.Style()
    style.configure('TNotebook.Tab', padding=(20, 8, 20, 0))
    self.notebook.pack(fill='both', expand='yes')

    ##########################################
    # Tabs
    ##########################################
    self.tab_seism  = ttk.Frame(self.notebook, padding=3, borderwidth=0)
    self.tab_seism.grid(row=0, column=0, sticky=(N, W, E, S))
    self.notebook.add(self.tab_seism, text='Seismic Modelling')

    self.tab_query   = ttk.Frame(self.notebook, padding=3, borderwidth=0)
    self.tab_query.grid(row=0, column=0, sticky=(W, E))
    self.notebook.add(self.tab_query, text='Query MESA')

    ##########################################
    # Mother & Child Frames
    ##########################################
    self.frame_conn = ttk.Labelframe(self.tab_seism, text='Connection to DB')
    self.frame_conn.grid(row=0, column=0, sticky=(W, E))

    self.frame_input = ttk.Labelframe(self.tab_seism, text='Observational Inputs')
    self.frame_input.grid(row=1, column=0, sticky=(W, E))

    self.frame_sample = ttk.Labelframe(self.tab_seism, text='Sampling')
    self.frame_sample.grid(row=2, column=0, sticky=(W, E))

    self.frame_MAP = ttk.Labelframe(self.tab_seism, text='Maximum a Posteriori')
    self.frame_MAP.grid(row=3, column=0, sticky=(W, E))

    self.frame_interp = ttk.Labelframe(self.tab_seism, text='Interpolation')
    self.frame_interp.grid(row=4, column=0, sticky=(W, E))

    self.frame_plot_modes = ttk.Labelframe(self.tab_seism, text='Observed Modes')
    self.frame_plot_modes.grid(row=0, column=1, sticky=(W, E, N, S), rowspan=5)

    self.frame_query = ttk.Labelframe(self.tab_query, text='Query MESA Models')
    self.frame_query.grid(row=0, column=0, sticky=(W, E))

    self.frame_stat_bar = ttk.Frame(root)
    self.frame_stat_bar.pack(side='top', fill='x', padx=4, pady=4)

    ##########################################
    # The connection frame
    ##########################################
    rad_but_loc       = ttk.Radiobutton(self.frame_conn, text='Local Disk', variable=self.connection, 
                                       value=self.conn_loc, command=self.connection_choice)
    rad_but_loc.grid(row=0, column=0)
    #
    rad_but_ivs       = ttk.Radiobutton(self.frame_conn, text='Inst. of Astron.', variable=self.connection,
                                       value=self.conn_ivs, command=self.connection_choice)
    rad_but_ivs.grid(row=0, column=1)
    #
    rad_but_https     = ttk.Radiobutton(self.frame_conn, text='HTTPS', variable=self.connection,
                                       value=self.conn_https, command=self.connection_choice)
    rad_but_https.grid(row=0, column=2)

    but_conn          = ttk.Button(self.frame_conn, text='Test Connection', command=self.check_connection) 
    but_conn.grid(row=1, column=0, sticky=W)

    self.conn_lbl     = ttk.Label(self.frame_conn, textvariable=self.conn_lbl_str, 
                                  relief='groove', anchor='center')
    self.conn_lbl_str.set('Inactive!')
    self.conn_lbl.grid(row=1, column=2, sticky=(W, E), rowspan=2)

    ##########################################
    # The Inputs Frame
    ##########################################
    self.but_input_freq = ttk.Button(self.frame_input, text='Load Frequency List', command=None) # self.browse_file
    self.but_input_freq.grid(row=0, column=0, sticky=(W, E))

    self.but_ex_freq    = ttk.Button(self.frame_input, text='Example')
    self.but_ex_freq.grid(row=0, column=1, sticky=(W, E))

    self.but_input_star = ttk.Button(self.frame_input, text='Load Global Parameters', command=None)
    self.but_input_star.grid(row=1, column=0, sticky=(W, E)) 

    self.but_input_star_ex = ttk.Button(self.frame_input, text='Example', command=None)
    self.but_input_star_ex.grid(row=1, column=1, sticky=(W, E))

    self.but_input_apply= ttk.Button(self.frame_input, text='Check & Input', command=None)
    self.but_input_apply.grid(row=2, column=0, sticky=(W, E))

    self.but_samp_input = ttk.Button(self.frame_sample, text='Load Sampling Settings', command=None)
    self.but_samp_input.grid(row=0, column=0, sticky=(W, E))

    self.but_samp_ex    = ttk.Button(self.frame_sample, text='Example', command=None)
    self.but_samp_ex.grid(row=0, column=1, sticky=(W, E))

    self.but_samp_exec  = ttk.Button(self.frame_sample, text='Execute', command=None)
    self.but_samp_exec.grid(row=0, column=2, sticky=(W, E))

    self.but_MAP        = ttk.Button(self.frame_MAP, text='Load MAP Settings', command=None)
    self.but_MAP.grid(row=0, column=0, sticky=(W, E))

    self.but_MAP_ex     = ttk.Button(self.frame_MAP, text='Example', command=None)
    self.but_MAP_ex.grid(row=0, column=1, sticky=(W, E))

    self.but_MAP_exec   = ttk.Button(self.frame_MAP, text='Execute', command=None)
    self.but_MAP_exec.grid(row=0, column=2, sticky=(W, E))

    self.but_interp     = ttk.Button(self.frame_interp, text='Load Interp. Settings', command=None)
    self.but_interp.grid(row=0, column=0, sticky=(W, E))

    self.but_interp_ex  = ttk.Button(self.frame_interp, text='Example', command=None)
    self.but_interp_ex.grid(row=0, column=1, sticky=(W, E))

    self.but_interp_exec=ttk.Button(self.frame_interp, text='Execute', command=None)
    self.but_interp_exec.grid(row=0, column=2, sticky=(W, E))

    self._=ttk.Label(self.frame_plot_modes, text='To Do ...')
    self._.grid(row=0, column=0)

    ##########################################
    # The Query Frame
    ##########################################
    self.but_query    = ttk.Button(self.frame_query, text='Load Query File', command=None)
    self.but_query.grid(row=0, column=0, sticky=(W, E))

    self.but_query_ex = ttk.Button(self.frame_query, text='Example', command=None)
    self.but_query_ex.grid(row=0, column=1, sticky=(W, E))

    self.but_query_exec = ttk.Button(self.frame_query, text='Execute', command=None)
    self.but_query_exec.grid(row=0, column=2, sticky=(W, E))

    self.save_query     = ttk.Checkbutton(self.frame_query, text='Save Results')
    self.save_query.grid(row=1, column=0, sticky=W)

    ##########################################
    # Display observed data
    ##########################################    
    # canv_input_freqs  = tk.Canvas(inputs_right, confine=False, cursor='circle', relief='flat')
    # coord             = 10, 50, 240, 240
    # arc               = canv_input_freqs.create_arc(coord, start=0, extent=150, fill="blue")
    # canv_input_freqs.pack(side='right')

  #   ##########################################
  #   # Sampling, and online plotting frame
  #   ##########################################
  #   Canv_plot_1 = tk.Canvas(self.frame_sample, confine=False)
  #   # plot_1_file = tk.PhotoImage(file=='Ehsan.JPG')
  #   # plot_1      = Canv_plot_1.create_image(500, 500, anchor='center', image=plot_1_file)
  #   # Canv_plot_1.pack()

  #   ##########################################
  #   # MAP analysis frame
  #   ##########################################
  #   but_MAP = tk.Button(self.frame_MAP, text='Max Likelihood')
  #   but_MAP.pack()

  #   # Checkboxes
  #   chk_var_1   = tk.IntVar()
  #   chk_1       = tk.Checkbutton(self.frame_MAP, text='Check this box if you like', command=None, 
  #                                justify='left', offvalue=False, onvalue=True,
  #                                selectcolor='green', state='active', variable=chk_var_1)
  #   chk_1.pack()
  #   chk_1.flash()

    ##########################################
    # Status Bar
    ##########################################
    # self.status_bar = ttk.Label(self.frame_stat_bar, bd=1, relief='flat', anchor='w')
    self.status_bar = ttk.Label(self.frame_stat_bar, relief='flat', anchor='w')
    self.status_bar.pack(side='bottom', fill='x')
    self.update_status_bar('')

    ##########################################

  # ##################################################################################
  # # M E T H O D S
  # ##################################################################################
  # # Connection 
  def connection_choice(self):
    """ Receive the 3 possible connection choices in int format """
    choice = self.connection.get()
    # self.connection.set(choice)
    if choice == self.conn_loc:   message = 'Connecting to the local machine (developer)'
    if choice == self.conn_ivs:   message = 'Connecting from inside the Institute of Astronomy (KULeuven)'
    if choice == self.conn_https: message = 'Connecting to a remote server via internet'
    self.update_status_bar(message)

  def check_connection(self):
    """ Try connecting to the database based on the user's choice """
    choice = self.connection.get()
    if choice not in [self.conn_loc, self.conn_ivs, self.conn_https]:
      self.update_status_bar('The connection choice is not made yet. Please choose one option.')
      self.conn_status = False
    else:    # attempt a connection test
      if choice == self.conn_loc: self.dbname   = self.conn_loc_str
      if choice == self.conn_ivs: self.dbname   = self.conn_ivs_str
      if choice == self.conn_https: self.dbname = self.conn_https_str
      self.do_connect()
    self.update_connection_state()

  def do_connect(self):
    """ The backend will connect based on the user's connecton choice """
    self.conn_status = bk.do_connect(self.dbname)

  def update_connection_state(self):
    """ Update the label in the connection frame based on the connection test """
    if self.conn_status:
      self.update_status_bar('Connection to {0} is active'.format(self.dbname))
      self.conn_lbl_str.set('Active')
      self.conn_lbl['background'] = 'green'
      self.conn_lbl['relief'] = 'sunken'
      # self.conn_lbl['image'] = self.handle_OK
    else:
      self.update_status_bar('The port {0} is unreachable'.format(self.dbname))
      self.conn_lbl_str.set('Unreachable')
      self.conn_lbl['background'] = 'red'
      self.conn_lbl['relief'] = 'raised'
      # self.conn_lbl['image'] = self.handle_NOK

  # ##########################################
  # # File Inputs
  # def browse_file(self):
  #   """ Browse the local disk for an ASCII file """
  #   fname = tkinter.filedialog.askopenfilename(title='Select Frequency List')
  #   try:
  #     bk.set_input_freq_file(fname)
  #     self.input_freq_file.set(fname)
  #     self.input_freq_done.set(True)
  #     self.update_status_bar('Successfully read the file: {0}'.format(fname))
  #   except:
  #     self.input_freq_done.set(False)
  #     self.update_status_bar('Failed to read the file: {0}'.format(fname))
    
  #   if self.input_freq_done:
  #     self.plot_observed_frequencies()

  # ##########################################
  # def example_input_freq(self):
  #   """ Show a static window with an example of how the input frequency file is structured """
  #   self.update_status_bar('Show an example of the input frequency list')
  #   ex_lines = bk.get_example_input_freq()
  #   new_wind = tk.Tk()
  #   new_wind.wm_title('Example: Frequency List')
  #   ex_lbl_  = tk.Label(new_wind, text=ex_lines, justify='left', cursor='arrow', relief='sunken',
  #                       padx=10, pady=10, wraplength=0)
  #   ex_lbl_.pack()
  #   new_wind.mainloop()

  # ##########################################
  # def plot_observed_frequencies(self):
  #   """ If reading in put frequencies is successfull, the modes will be plotted, as a reward! """

  #   fig       = matplotlib.figure.Figure(figsize=(5, 5), dpi=100, tight_layout=True)
  #   ax        = fig.add_subplot(111)

  #   modes     = bk.bk_star.get('modes')
  #   freqs     = np.array([ mode.freq for mode in modes ])
  #   freq_errs = np.array([ mode.freq_err for mode in modes ])
  #   per       = old_div(1.,freqs)
  #   per_err   = old_div(freq_errs, freqs**2)
  #   arr_l     = np.array([ mode.l for mode in modes ])
  #   arr_m     = np.array([ mode.m for mode in modes ])
  #   arr_p_mode= np.array([ mode.p_mode for mode in modes ])
  #   arr_g_mode= np.array([ mode.g_mode for mode in modes ])
  #   arr_in_df = np.array([ mode.in_df for mode in modes ])
  #   arr_in_dP = np.array([ mode.in_dP for mode in modes ])

  #   ax.plot(freqs, np.zeros(len(freqs)))

  #   ax.set_xlabel('Frequency [per day]')

  #   canvas = FigureCanvasTkAgg(fig, master=self.inputs_right)
  #   canvas.show()

  #   canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
  #   canvas._tkcanvas.pack(side='top', fill='both', expand=True)

  # ##########################################
  # # Observatinal Inputs
  # def release_obs_log_Teff(self):
  #   """ Release the log_Teff two entry boxes """
  #   choice = self.use_obs_log_Teff.get()
  #   if choice:
  #     self.enter_log_Teff['state'] = 'normal'
  #     self.enter_log_Teff_err['state'] = 'normal'

  # ##########################################
  # def release_obs_log_g(self):
  #   """ Release the log_g two entry boxes """
  #   choice = self.use_obs_log_g.get()
  #   if choice:
  #     self.enter_log_g['state'] = 'normal'
  #     self.enter_log_g_err['state'] = 'normal'

  # ##########################################
  # def get_observations(self):
  #   """ Get all the observed values which are set in Entry boxes """
  #   # log(Teff)
  #   if self.use_obs_log_Teff.get():
  #     val = float(self.enter_log_Teff.get())
  #     err = float(self.enter_log_Teff_err.get())
  #     val_OK = False
  #     if not (3 <= val <= 5):
  #       self.enter_log_Teff['bg'] = 'red'
  #       self.update_status_bar('log(Teff)={0} is regjected. Only 3 <= log(Teff) <= 5 is accepted'.format(val) )
  #     else:
  #       val_OK = True
  #       self.enter_log_Teff['bg'] = 'white'
  #       self.obs_log_Teff.set(val)
      
  #     err_OK = False
  #     if not (0 <= err <= 4):
  #       self.enter_log_Teff_err['bg'] = 'red'
  #       self.update_status_bar('err[log(Teff)]={0} is rejected. Only 0 <= err <= 4 is accepted'.format(err))
  #     else:
  #       err_OK = True
  #       self.enter_log_Teff_err['bg'] = 'white'
  #       self.obs_log_Teff_err.set(err)

  #     if val_OK and err_OK:        
  #       self.update_status_bar('log(Teff) = {0} +/- {1}'.format(val, err))
  #       bk.set_obs_log_Teff(val, err)

  #   # log(g)
  #   if self.use_obs_log_g.get():
  #     val = float(self.enter_log_g.get())
  #     err = float(self.enter_log_g_err.get())
  #     val_OK = False
  #     if not (0 <= val <= 4.5):
  #       self.enter_log_g['bg'] = 'red'
  #       self.update_status_bar('log(g)={0} is rejected. Only 0 <= log(g) <= 4.5 is accepted'.format(val))
  #     else:
  #       val_OK = True 
  #       self.enter_log_g['bg'] = 'white'
  #       self.obs_log_g.set(val)

  #     err_OK = False
  #     if not (0 <= err <= 1):
  #       self.enter_log_g_err['bg'] = 'red'
  #       self.update_status_bar('err[log(g)]={0} is rejected. Only 0 <= err <= 1 is accepted.'.format(err))
  #     else:
  #       err_OK = True 
  #       self.enter_log_g_err['bg'] = 'white'
  #       self.obs_log_g_err.set(err)

  #     if val_OK and err_OK:
  #       self.update_status_bar('log(g) = {0} +/- {1}'.format(val, err))
  #       bk.set_obs_log_g(val, err)

  # ##########################################
  # # Inputs through checkbuttons and radiobuttons
  # def sampling_setup(self):
  #   if self.which_sampling_function == self.sampling_constrained:
  #     self.update_status_bar('Learning set will be selected with constraints (on log_Teff, log_g, eta)')
  #   elif self.which_sampling_function == self.sampling_randomly:
  #     self.update_status_bar('Learning set will be randomly selected')
  #   bk.set_sampling_function(self.which_sampling_function.get())
    
  # def shuffling_setup(self):
  #   if self.sampling_shuffle:
  #     self.update_status_bar('Randomly shuffle the learning set')
  #   bk.set_shuffling(self.sampling_shuffle.get())

  # ##########################################
  # Status Bar
  def update_status_bar(self, message):
    """ Concurrently update the status printed on the StatusBar. Very handly method here! """
    self.status_bar_message = 'Status: {0}'.format(message)
    self.status_bar['text'] = self.status_bar_message

####################################################################################
if __name__ == '__main__':
  master    = Tk()         # invoke the master frame
  session   = GUI(master)  # instantiate the GUI
  master.mainloop()        # keep it alive, until terminated/closed

####################################################################################

