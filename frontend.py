
import sys
import Tkinter as tk 
from grid import backend as bk

####################################################################################
##########################################
# The mother frame and the children frames
##########################################
root = tk.Tk()
frame_conn = tk.Frame(root, highlightbackground='grey', highlightcolor='grey', highlightthickness=1, bd=0)
frame_conn.pack(side='top', expand=True, fill='both')

frame_inputs = tk.Frame(root, highlightbackground='grey', highlightcolor='grey', highlightthickness=1, bd=0)
frame_inputs.pack(side='top', expand=True, fill='both')

frame_sample = tk.Frame(root)
frame_sample.pack(side='top')

frame_MAP = tk.Frame(root)
frame_MAP.pack(side='top')

##########################################
# The connection frame
##########################################
# define the left, middle and right insets
conn_left         = tk.Frame(frame_conn, highlightbackground='grey', highlightcolor='grey', 
                             highlightthickness=1, bd=0)
conn_left.pack(side='left')
conn_middle       = tk.Frame(frame_conn, highlightbackground='grey', highlightcolor='grey', 
                             highlightthickness=1, bd=0)
conn_middle.pack(side='left')
conn_right        = tk.Frame(frame_conn, highlightbackground='grey', highlightcolor='grey', 
                             highlightthickness=1, bd=0)
conn_right.pack(side='left')

lbl_choice_var    = tk.StringVar()
lbl_choice        = tk.Label(conn_left, textvariable=lbl_choice_var, relief='flat')
lbl_choice_var.set('Choose: ')
lbl_choice.pack(side='left')

# Put stuff on the insets now
rad_but_var       = tk.IntVar()
rad_but_loc_val   = 1   # 'grid'
rad_but_ivs_val   = 2   # Institute of Astronomy
rad_but_https_val = 3  # 'https://'
rad_but_loc       = tk.Radiobutton(conn_left, text='Local Disk', variable=rad_but_var, 
                                   value=rad_but_loc_val, command=None)
rad_but_loc.pack(side='left')
rad_but_ivs       = tk.Radiobutton(conn_left, text='Inst. of Astron.', variable=rad_but_var,
                                   value=rad_but_ivs_val, command=None)
rad_but_ivs.pack(side='left')
rad_but_https     = tk.Radiobutton(conn_left, text='HTTPS', variable=rad_but_var,
                                   value=rad_but_https_val, command=None)
rad_but_https.pack(side='left')

# txt_result        = tk.Text(frame_conn, exportselection=0)
# txt_result.insert('insert', 'Result: ')
# txt_result.pack(side='right')

but_conn          = tk.Button(conn_middle, text='Test Connection', command=None)
but_conn.pack(side='right')

lbl_status_var    = tk.StringVar()
lbl_status        = tk.Label(conn_right, textvariable=lbl_status_var, relief='groove', command=None)
lbl_status_var.set('Inactive!')
lbl_status.pack(side='right')

lbl_conn_var      = tk.StringVar()
lbl_conn          = tk.Label(conn_right, textvariable=lbl_conn_var, relief='flat')
lbl_conn_var.set('Result: ')
lbl_conn.pack(side='right')

##########################################
# The inputs frame
##########################################
# define the left and right insets
inputs_left       = tk.Frame(frame_inputs, highlightbackground='grey', highlightcolor='grey', 
                             highlightthickness=1, bd=0)
inputs_left.pack(expand=True, side='left', fill='both')
inputs_right      = tk.Frame(frame_inputs, highlightbackground='grey', highlightcolor='grey', 
                             highlightthickness=1, bd=0)
inputs_right.pack(expand=True, side='right', fill='both')

# load file buttons
but_input_freq    = tk.Button(inputs_left, text='Load Frequency List')
but_input_freq.pack(side='top')
but_ex_freq       = tk.Button(inputs_left, text='Example Frequency List')
but_ex_freq.pack(side='top')
but_ex_options    = tk.Button(inputs_left, text='Example setting List')
but_ex_options.pack(side='bottom')
but_input_options = tk.Button(inputs_left, text='Load Setting List')
but_input_options.pack(side='bottom')

# plots
canv_input_freqs  = tk.Canvas(inputs_right, confine=False, cursor='circle', relief='flat')
coord             = 10, 50, 240, 240
arc               = canv_input_freqs.create_arc(coord, start=0, extent=150, fill="blue")
canv_input_freqs.pack(side='right')


##########################################
# Sampling, and online plotting frame
##########################################
Canv_plot_1 = tk.Canvas(frame_sample, confine=False)
plot_1_file = tk.PhotoImage(file=='Ehsan.JPG')
plot_1      = Canv_plot_1.create_image(500, 500, anchor='center', image=plot_1_file)
Canv_plot_1.pack()

##########################################
# MAP analysis frame
##########################################
but_MAP = tk.Button(frame_MAP, text='Max Likelihood')
but_MAP.pack()

# Checkboxes
chk_var_1   = tk.IntVar()
chk_1       = tk.Checkbutton(frame_MAP, text='Check this box if you like', command=None, 
                             justify='left', offvalue=False, onvalue=True,
                             selectcolor='green', state='active', variable=chk_var_1)
chk_1.pack()
chk_1.flash()

##########################################
# The main loop to keep the GUI alive
##########################################
root.mainloop()
####################################################################################

