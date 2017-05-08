#! /usr/bin/python

"""
This module is the user's frontend to open a graphical user interface, GUI, and interact with the 
underlying grid database and a (limited) suite of functionalities in the backend. There are millions 
of ways one can design the GUI, provide different functionalities, and/or allow the user to interact 
with the database. What is provided here is one way out of the millions of possibilities, and is a 
basic attempt to start with the GUI. Indeed, this GUI is pretty extensible, and flexible to grow and
accommodate additional tools for analysis.
"""
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
    self.root=root          # The only input argument

    # Global variables
    self.dir_data     = 'data/'
    self.dir_bitmaps  = self.dir_data + 'bitmaps/'

    # All Tabs
    self.tab_inputs     = 0
    self.tab_sampling   = 0
    self.tab_ann        = 0
    self.tab_interp     = 0

    self.mainframe      = 0

    # All useful frames and insets
    self.frame_conn     = 0
    self.frame_inputs   = 0
    self.frame_sample   = 0
    self.frame_MAP      = 0
    self.frame_stat_bar = 0

    self.inputs_left    = 0
    self.inputs_right   = 0

    # Connection constant values
    self.connection   = IntVar()
    self.conn_loc     = 1   # Development, accessing the 'grid' database
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

    # Input frequency list
    self.input_freq_file = StringVar()
    self.input_freq_done = BooleanVar()

    # Input observational data
    self.use_obs_log_Teff = BooleanVar()
    self.obs_log_Teff     = DoubleVar()
    self.obs_log_Teff_err = DoubleVar()

    self.use_obs_log_g    = BooleanVar()
    self.obs_log_g        = DoubleVar()
    self.obs_log_g_err    = DoubleVar()

    # Sampling 
    self.which_sampling_function = BooleanVar()
    self.sampling_constrained    = True 
    self.sampling_randomly       = False
    self.sampling_shuffle        = BooleanVar()        

    # Status bar messages
    self.status_bar_message = StringVar()

    # Trigger the FrontWindow now ...
    self.FrontWindow()      

  def FrontWindow(self):
    """ The front GUI window """

    root       = self.root

    root.wm_title('Asteroseismic Modelling Tool')
    root.title('The 2nd title!')

    ##########################################
    # The mother frame and the children frames
    ##########################################
    self.mainframe  = ttk.Frame(root, padding='3 3 3 3')
    self.mainframe.grid(row=0, column=0, sticky=(N, W, E, S))
    self.mainframe.columnconfigure(0, weight=1)
    self.mainframe.rowconfigure(0, weight=1)

  #   self.frame_conn = tk.Frame(root, highlightbackground='grey', highlightcolor='grey', highlightthickness=1, bd=0)
  #   self.frame_conn.pack(side='top', expand=True, fill='both', padx=10, pady=10)

  #   self.frame_inputs = tk.Frame(root, highlightbackground='grey', highlightcolor='grey', highlightthickness=1, bd=0)
  #   self.frame_inputs.pack(side='top', expand=True, fill='both')

  #   self.frame_sample = tk.Frame(root)
  #   self.frame_sample.pack(side='top')

  #   self.frame_MAP = tk.Frame(root)
  #   self.frame_MAP.pack(side='top')

  #   self.frame_stat_bar = tk.Frame(root, highlightbackground='grey', highlightcolor='grey', highlightthickness=1, bd=0)
  #   self.frame_stat_bar.pack(side='top', fill='x', padx=4, pady=4)

  #   ##########################################
  #   # The connection frame
  #   ##########################################
  #   # define the left, middle and right insets
  #   conn_left         = tk.Frame(self.frame_conn, padx=20, pady=10)
  #   conn_left.pack(side='left')
  #   conn_middle       = tk.Frame(self.frame_conn, padx=20, pady=10)
  #   conn_middle.pack(side='left')
  #   conn_right        = tk.Frame(self.frame_conn, padx=20, pady=10)
  #   conn_right.pack(side='left')

  #   lbl_choice_var    = tk.StringVar()
  #   lbl_choice        = tk.Label(conn_left, textvariable=lbl_choice_var, relief='flat')
  #   lbl_choice_var.set('Choose: ')
  #   lbl_choice.pack(side='left')

  #   # Put stuff on the insets now
  #   rad_but_loc       = tk.Radiobutton(conn_left, text='Local Disk', variable=self.connection, 
  #                                      value=self.conn_loc, command=self.connection_choice)
  #   rad_but_loc.pack(side='left')
  #   #
  #   rad_but_ivs       = tk.Radiobutton(conn_left, text='Inst. of Astron.', variable=self.connection,
  #                                      value=self.conn_ivs, command=self.connection_choice)
  #   rad_but_ivs.pack(side='left')
  #   #
  #   rad_but_https     = tk.Radiobutton(conn_left, text='HTTPS', variable=self.connection,
  #                                      value=self.conn_https, command=self.connection_choice)
  #   rad_but_https.pack(side='left')

  #   # txt_result        = tk.Text(frame_conn, exportselection=0)
  #   # txt_result.insert('insert', 'Result: ')
  #   # txt_result.pack(side='right')

  #   but_conn          = tk.Button(conn_middle, text='Test Connection', command=self.check_connection) 
  #   but_conn.pack(side='right')

  #   self.conn_lbl     = tk.Label(conn_right, textvariable=self.conn_lbl_str, relief='groove')
  #   self.conn_lbl_str.set('Inactive!')
  #   self.conn_lbl.pack(side='right')

  #   lbl_conn_var      = tk.StringVar()
  #   lbl_conn          = tk.Label(conn_right, textvariable=lbl_conn_var, relief='flat')
  #   lbl_conn_var.set('Result: ')
  #   lbl_conn.pack(side='right')

  #   ##########################################
  #   # The inputs frame
  #   ##########################################
  #   # define the left and right insets
  #   self.inputs_left       = tk.Frame(self.frame_inputs, highlightbackground='grey', highlightcolor='grey', 
  #                                highlightthickness=1, bd=0)
  #   self.inputs_left.pack(expand=True, side='left', fill='both')
  #   self.inputs_right      = tk.Frame(self.frame_inputs, highlightbackground='grey', highlightcolor='grey', 
  #                                highlightthickness=1, bd=0)
  #   self.inputs_right.pack(expand=True, side='right', fill='both')
  #   i_rows            = cycle(list(range(100)))

  #   # The observational inputs
  #   # load file buttons
  #   i_row                  = next(i_rows)
  #   self.but_input_freq    = tk.Button(self.inputs_left, text='Load Frequency List', command=self.browse_file)
  #   self.but_input_freq.grid(row=i_row, column=0, sticky='w')
  #   self.but_ex_freq       = tk.Button(self.inputs_left, text='Example', command=self.example_input_freq)
  #   self.but_ex_freq.grid(row=i_row, column=1)
  #   self.but_input_options = tk.Button(self.inputs_left, text='Load Setting List')
    
  #   # log(Teff)
  #   i_row                  = next(i_rows)
  #   self.ckbx_obs_log_Teff = tk.Checkbutton(self.inputs_left, text='Use observed log(Teff) +/- err [K]', 
  #                                           variable=self.use_obs_log_Teff, command=self.release_obs_log_Teff, 
  #                                           offvalue=False, onvalue=True, justify='left')
  #   self.ckbx_obs_log_Teff.grid(row=i_row, sticky='w')

  #   i_row                  = next(i_rows) # next row
  #   frame_log_Teff         = tk.Frame(self.inputs_left) # create a local frame to pack 4 items in it
  #   frame_log_Teff.grid(row=i_row, columnspan=2, sticky='w')

  #   self.enter_log_Teff    = tk.Entry(frame_log_Teff, justify='left', textvariable=self.obs_log_Teff, 
  #                                     state='disabled', width=8)
  #   self.enter_log_Teff.grid(row=i_row, column=0, sticky='w', padx=5)
  #   self.enter_log_Teff_err= tk.Entry(self.inputs_left, justify='left', textvariable=self.obs_log_Teff_err, 
  #                                     state='disabled', width=6)
  #   self.enter_log_Teff_err.grid(row=i_row, column=1, sticky='w', padx=5)

  #   # log(g)
  #   i_row                  = next(i_rows)
  #   self.ckbx_obs_log_g    = tk.Checkbutton(self.inputs_left, text='Use observed log(g) +/- err [cgs]', 
  #                                           variable=self.use_obs_log_g, command=self.release_obs_log_g, 
  #                                           offvalue=False, onvalue=True, justify='left')
  #   self.ckbx_obs_log_g.grid(row=i_row, sticky='w')

  #   i_row                  = next(i_rows) # next row
  #   frame_log_g            = tk.Frame(self.inputs_left) # create a local frame to pack 4 items in it
  #   frame_log_g.grid(row=i_row, columnspan=2, sticky='w')

  #   self.enter_log_g       = tk.Entry(frame_log_g, justify='left', state='disabled', width=8)
  #   self.enter_log_g.grid(row=i_row, column=0, sticky='w', padx=5)
  #   self.enter_log_g_err= tk.Entry(self.inputs_left, justify='left', state='disabled', width=6)
  #   self.enter_log_g_err.grid(row=i_row, column=1, sticky='w', padx=5)

  #   # Button to set observations
  #   i_row                  = next(i_rows)
  #   self.but_obs           = tk.Button(self.inputs_left, text='Set Obs. Values', command=self.get_observations)
  #   self.but_obs.grid(row=i_row, sticky='e')

  #   # The modeling settings
  #   i_row                  = next(i_rows)
  #   separator = tk.Frame(self.inputs_left, height=2, bd=1, relief='sunken')
  #   separator.grid(row=i_row, sticky='ew', columnspan=2)
  #   i_row                  = next(i_rows)
  #   self.but_input_options.grid(row=i_row, column=0, sticky='w')
  #   self.but_ex_options    = tk.Button(self.inputs_left, text='Example')
  #   self.but_ex_options.grid(row=i_row, column=1)

  #   # Options to control behaviour/analysis
  #   i_row                  = next(i_rows)
  #   self.rad_but_samp_cnstr= tk.Radiobutton(self.inputs_left, text='Sample with Constraints', 
  #                                           variable=self.which_sampling_function, 
  #                                           value=self.sampling_constrained, 
  #                                           command=self.sampling_setup)
  #   self.rad_but_samp_cnstr.grid(row=i_row)
  #   self.rad_but_samp_rand = tk.Radiobutton(self.inputs_left, text='Sample Randomly', 
  #                                           variable=self.which_sampling_function,
  #                                           value=self.sampling_randomly, 
  #                                           command=self.sampling_setup)
  #   #
  #   i_row                  = next(i_rows)
  #   self.rad_but_samp_rand.grid(row=i_row, sticky='w')
  #   self.ckbx_sampling_shuffle = tk.Checkbutton(self.inputs_left, text='Shuffle the learning set', 
  #                                           variable=self.sampling_shuffle, command=self.shuffling_setup, 
  #                                           offvalue=False, onvalue=True, justify='left')
  #   #
  #   i_row                  = next(i_rows)
  #   self.ckbx_sampling_shuffle.grid(row=i_row, sticky='w')
  #   i_row                  = next(i_rows)
  #   separator = tk.Frame(self.inputs_left, height=2, bd=1, relief='sunken')
  #   separator.grid(row=i_row, sticky='ew', columnspan=2)

  #   ##########################################
  #   # Display observed data
  #   ##########################################    
  #   # canv_input_freqs  = tk.Canvas(inputs_right, confine=False, cursor='circle', relief='flat')
  #   # coord             = 10, 50, 240, 240
  #   # arc               = canv_input_freqs.create_arc(coord, start=0, extent=150, fill="blue")
  #   # canv_input_freqs.pack(side='right')

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

  #   ##########################################
  #   # Status Bar
  #   ##########################################
  #   self.status_bar = tk.Label(self.frame_stat_bar, bd=1, relief='flat', anchor='w')
  #   self.status_bar.pack(side='bottom', fill='x')
  #   self.update_status_bar('')

  #   ##########################################

  # ##################################################################################
  # # M E T H O D S
  # ##################################################################################
  # # Connection 
  # def connection_choice(self):
  #   """ Receive the 3 possible connection choices in int format """
  #   choice = self.connection.get()
  #   self.connection.set(choice)
  #   if choice == self.conn_loc:   message = 'Connecting to the local machine (developer)'
  #   if choice == self.conn_ivs:   message = 'Connecting from inside the Institute of Astronomy (KULeuven)'
  #   if choice == self.conn_https: message = 'Connecting to a remote server via internet'
  #   self.update_status_bar(message)

  # def check_connection(self):
  #   """ Try connecting to the database based on the user's choice """
  #   choice = self.connection.get()
  #   if choice not in [self.conn_loc, self.conn_ivs, self.conn_https]:
  #     self.update_status_bar('The connection choice is not made yet. Please choose one option.')
  #     self.conn_status = False
  #   else:    # attempt a connection test
  #     if choice == self.conn_loc: self.dbname   = self.conn_loc_str
  #     if choice == self.conn_ivs: self.dbname   = self.conn_ivs_str
  #     if choice == self.conn_https: self.dbname = self.conn_https_str
  #     self.do_connect()
  #   self.update_connection_state()

  # def do_connect(self):
  #   """ The backend will connect based on the user's connecton choice """
  #   self.conn_status = bk.do_connect(self.dbname)

  # def update_connection_state(self):
  #   """ Update the label in the connection frame based on the connection test """
  #   if self.conn_status:
  #     self.update_status_bar('Connection to {0} is active'.format(self.dbname))
  #     self.conn_lbl_str.set('Active')
  #     self.conn_lbl['bg'] = 'green'
  #     self.conn_lbl['relief'] = 'sunken'
  #     # self.conn_lbl['image'] = self.handle_OK
  #   else:
  #     self.update_status_bar('The port {0} is unreachable'.format(self.dbname))
  #     self.conn_lbl_str.set('Unreachable')
  #     self.conn_lbl['bg'] = 'red'
  #     self.conn_lbl['relief'] = 'raised'
  #     # self.conn_lbl['image'] = self.handle_NOK

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
  # # Status Bar
  # def update_status_bar(self, message):
  #   """ Concurrently update the status printed on the StatusBar. Very handly method here! """
  #   self.status_bar_message = 'Status: {0}'.format(message)
  #   self.status_bar['text'] = self.status_bar_message

####################################################################################
if __name__ == '__main__':
  master    = Tk()      # invoke the master frame
  session   = GUI(master)  # instantiate the GUI
  master.mainloop()        # keep it alive, until terminated/closed

####################################################################################

