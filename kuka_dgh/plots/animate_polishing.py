'''
Dynamic force plot from mds 
'''
from mim_data_utils import DataReader
import numpy as np
import matplotlib.pyplot as plt 

import pathlib
import os
python_path = pathlib.Path('.').absolute().parent
os.sys.path.insert(1, str(python_path))
print(python_path)
from utils import path_utils
from plot_utils import SimpleDataPlotter
from utils.reduced_model import get_controlled_joint_ids
from utils import analysis_utils

from croco_mpc_utils.pinocchio_utils import *
from mim_robots.robot_loader import load_pinocchio_wrapper


import matplotlib
matplotlib.use('GTK3Agg') 


from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
import time


# Load robot model
LOCKED_JOINTS = ['A7']
pinrobot = load_pinocchio_wrapper('iiwa_ft_sensor_shell', locked_joints=LOCKED_JOINTS)
model    = pinrobot.model
data     = model.createData()
frameId  = model.getFrameId('contact')
nq = model.nq ; nv = model.nv ; nc = 1
# Overwrite effort limit as in DGM
model.effortLimit = np.array([100, 100, 50, 50, 20, 10, 10])
controlled_joint_ids = get_controlled_joint_ids('iiwa_ft_sensor_shell', locked_joints=LOCKED_JOINTS)

# Load config file
CONFIG_NAME = 'polishing'
CONFIG_PATH = "config/"+CONFIG_NAME+".yml"
config      = path_utils.load_yaml_file(CONFIG_PATH)



# Load data 
SIM = False

# Create data Plottger
s = SimpleDataPlotter()


FILTER = 20
data_path = '/home/skleff/Videos/classical_mpc_coulomb/'


print("Load data 1...")
r1 = DataReader(data_path+'polishing_REAL_2024-02-09T12_06_59.991959_coulomb_medium.mds') 

label1 = 'Classical MPC'
 

PART = 1
N_TOT = r1.data['tau'].shape[0]
T = 6283 * 3 / config['plan_freq']
time_lin1 = r1.data['absolute_time'] 
N_START = np.sum(time_lin1 < config['T_CIRCLE'])
N_END = np.sum(time_lin1 <= T + config['T_CIRCLE'])
print("T       = ", T)
print("N_TOT   = ", N_TOT)
print("N_START = ", N_START)
print("N_END = ", N_END)
print("----")


# Split trajectory (skip some frames) to reduce animation size 
SPLIT = 10
print("SPLIT = ", SPLIT)
time_lin1_split = time_lin1[N_START:N_END:SPLIT] - config['T_CIRCLE']

# Extract measured signals of force and position
force_1_split = r1.data['contact_force_3d_measured'][N_START:N_END:SPLIT, 2:3]
p_mea1_split = get_p_(r1.data['joint_positions'][N_START:N_END:SPLIT,controlled_joint_ids][:,controlled_joint_ids], pinrobot.model, pinrobot.model.getFrameId('contact'))

if(FILTER > 0):
    force_1_split = analysis_utils.moving_average_filter(force_1_split.copy(), FILTER)


# Extract reference signals of force and position
target_force = np.zeros(time_lin1_split.shape)
target_force[:] = config['frameForceRef'][2]
target_position_1 = np.zeros((N_TOT, 3))
target_position_1[:,0] = r1.data['target_position_x'][:N_TOT, 0]
target_position_1[:,1] = r1.data['target_position_y'][:N_TOT, 0]
target_position_1[:,2] = r1.data['target_position_z'][:N_TOT, 0]
target_position_split = target_position_1[N_START:N_END:SPLIT]
err1_split = np.linalg.norm(p_mea1_split[:, :2] - target_position_split[:, :2], axis=1)*1e3

print("PLOTTING")

color_list = ['b', 'g', 'r', 'y']

WITH_POSITION_PLOT = False
if(WITH_POSITION_PLOT):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(68, 16))
else:
    fig, ax1 = plt.subplots(1, 1, sharex='col', figsize=(68, 16))
fig.canvas.draw() 

ax1.plot(time_lin1_split, target_force, color='k', linewidth=10, linestyle='--', label='Reference', alpha=0.6) 
ax1.grid(linewidth=1)
ax1.set_xlim(time_lin1_split[0], time_lin1_split[-1])
ax1.set_ylim(34., 60.)
ax1.set_ylabel('Normal force (N)', fontsize=48)
ax1.tick_params(axis = 'y', labelsize=48)
ax1.tick_params(labelbottom=False)  

# Ax 2 (position)
if(WITH_POSITION_PLOT):
    ax2.plot(time_lin1_split, np.zeros_like(err1_split), linestyle='--', marker=None, color='k', linewidth=10, alpha=0.6, label='Reference')
    ax2.grid(linewidth=1)
    ax2.set_xlim(time_lin1_split[0], time_lin1_split[-1])
    ax2.set_ylim(-1., 50.)
    ax2.set_ylabel('Position error (mm)', fontsize=48)
    ax2.set_xlabel('Time (s)', fontsize=48)
    ax2.tick_params(axis = 'x', labelsize=48)
    ax2.tick_params(axis = 'y', labelsize=48)
    ax2.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

if PART == 1:
    line_f_r1, = ax1.plot(time_lin1_split[0:1], force_1_split[0:1], animated=True, color='b', linewidth=10, label=label1, alpha=0.9)

    if(WITH_POSITION_PLOT):
        line_p_r1, = ax2.plot(time_lin1_split[0:1], err1_split[0:1], animated=True, color='b', linewidth=10, label=label1, alpha=0.9)
        line = [line_f_r1, line_p_r1]
    else:
        line = [line_f_r1]
        
# if PART == 2:
#     line_f_r1, = ax1.plot(time_lin_1[:], force_1[:], animated=True, color=color_list[1], linewidth=6, label=label1, alpha=0.8)
#     line_f_r2, = ax1.plot(time_lin_2[0:1], force_2[0:1], animated=True, color=color_list[2], linewidth=6, label=label2, alpha=0.8)


#     line_df_r1, = ax2.plot(time_lin_2[:], np.zeros(time_lin_2.shape), animated=True, color=color_list[1], linewidth=6, label=label1, alpha=0.8)
#     line_df_r2, = ax2.plot(time_lin_2[0:1], delta_F[0:1], animated=True, color=color_list[2], linewidth=6, label=label2, alpha=0.8)
#     line = [line_f_r1, line_df_r1, line_f_r2, line_df_r2]

# Task phases
# PHASE_TIME = 6283
# ax1.axvline(PHASE_TIME/1000., color='k', linewidth=4, linestyle='-', alpha=1.)
# ax1.axvline(2*PHASE_TIME/1000., color='k', linewidth=4, linestyle='-', alpha=1.)
# ax2.axvline(PHASE_TIME/1000., color='k', linewidth=4, linestyle='-', alpha=1.)
# ax2.axvline(2*PHASE_TIME/1000., color='k', linewidth=4, linestyle='-', alpha=1.)
# fig.text(0.22, 0.90, 'Slow', transform=fig.transFigure, fontdict={'size':70})
# fig.text(0.5, 0.90, 'Medium',transform=fig.transFigure, fontdict={'size':70})
# fig.text(0.82, 0.90, 'Fast',transform=fig.transFigure, fontdict={'size':70})

# if PART == 1:
#     fig.text(0.48, 0.3, 'No estimation',transform=fig.transFigure, fontdict={'size':70})

ax1.legend(loc="upper right", framealpha=0.95, fontsize=40) 

fig.align_ylabels()
fig.tight_layout(pad=1)


PPS = 100  # Point per second

N_FRAMES = int(T * PPS)
SKIP = int(1000/PPS)




def init():
    """
    This init function defines the initial plot parameter
    """
    # Set initial parameter for the plot
    return line

def animate(t):
    """
    This function will be called periodically by FuncAnimation. Frame parameter will be passed on each call as a counter. 
    """
    if PART == 1:
        mask1 = time_lin1_split < t
        line[0].set_data(time_lin1_split[mask1], force_1_split[mask1])
        if(WITH_POSITION_PLOT):
            line[1].set_data(time_lin1_split[mask1], err1_split[mask1.squeeze()])

    # if PART == 2:
    #     mask2 = time_lin_2 < t
    #     line[2].set_data(time_lin_2[mask2], force_2[mask2])
    #     line[3].set_data(time_lin_2[mask2], delta_F[mask2])


    return line

# Create FuncAnimation object and plt.show() to show the updated animation


t0 = time.time()

time_lin = np.linspace(0, T, N_FRAMES)
ani = FuncAnimation(fig, animate, frames=time_lin, repeat=False, interval = SKIP, init_func = init, blit=True)
folder = data_path 

ani.save(folder + 'polishing_coulomb_animation_NEW' + '.mp4')

print("COMPUTE TIME = ", time.time() - t0)



