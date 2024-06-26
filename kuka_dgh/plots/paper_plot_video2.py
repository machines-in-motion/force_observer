from mim_data_utils import DataReader
import numpy as np
import matplotlib.pyplot as plt 

import pathlib
import os
python_path = pathlib.Path('.').absolute().parent
os.sys.path.insert(1, str(python_path))
print(python_path)
from utils import path_utils, pin_utils
from plot_utils import SimpleDataPlotter
from utils.reduced_model import get_controlled_joint_ids
from utils import analysis_utils

import croco_mpc_utils.pinocchio_utils as pin_utils
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


FILTER = 400
data_path = '/home/skleff/Desktop/ICRA24/dataset_force_offset_ICRA_2024/sanding/friction_compensation/'


print("Load data 1...")
r1 = DataReader(data_path+'config_REAL_2023-09-13T19_42_59.978248_medium_coulomb.mds') 
print("Load data 2...")
r2 = r1 
# r2 = DataReader(data_path+'config_REAL_2023-09-21T16:09:53.109844_FL_DF_PM_perturbation_2.mds') 


label1 = 'FL'
label2 = r'FL + $\Delta F$'
 

PART = 1


N_TOT = min(r1.data['tau'].shape[0], r2.data['tau'].shape[0])

T = 6283 * 3 / config['plan_freq']

time_lin1 = np.linspace(0, N_TOT/ config['plan_freq'], (N_TOT)) #r1.data['time']
time_lin2 = np.linspace(0, N_TOT/ config['plan_freq'], (N_TOT)) #r2.data['time']

N_START_1 = np.sum(time_lin1 < config['T_CIRCLE'])
N_START_2 = np.sum(time_lin2 < config['T_CIRCLE'])

N_END_1 = np.sum(time_lin1 <= T + config['T_CIRCLE'])
N_END_2 = np.sum(time_lin2 <= T + config['T_CIRCLE'])

if(FILTER > 0):
    print("FILTERING")
    force_1 = analysis_utils.moving_average_filter(r1.data['contact_force_3d_measured'][N_START_1:N_END_1, 2:3].copy(), FILTER)
    force_2 = analysis_utils.moving_average_filter(r2.data['contact_force_3d_measured'][N_START_2:N_END_2, 2:3].copy(), FILTER) 
else:
    force_1 = r1.data['contact_force_3d_measured'][N_START_1:N_END_1, 2:3]
    force_2 = r2.data['contact_force_3d_measured'][N_START_2:N_END_2, 2:3] 


delta_F =  analysis_utils.moving_average_filter(r2.data['delta_f'][N_START_2:N_END_2].copy(), 100) 



SPLIT = 10
time_lin_1 = time_lin1[N_START_1:N_END_1:SPLIT] - config['T_CIRCLE']
time_lin_2 = time_lin2[N_START_2:N_END_2:SPLIT] - config['T_CIRCLE']


force_1 = force_1[::SPLIT]
force_2 = force_2[::SPLIT]

delta_F = delta_F[::SPLIT]

target_force = np.zeros(time_lin_1.shape)
target_force[:] = config['frameForceRef'][2]


color_list = ['b', 'g', 'r', 'y']


print("PLOTTING")

fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(68, 16))

fig.canvas.draw() 

ax1.plot(time_lin_1, target_force, color='k', linewidth=6, linestyle='--', label='Reference', alpha=0.5) 
ax1.grid(linewidth=1)
ax2.grid(linewidth=1) 
ax2.set_ylim(-20, 40)
ax1.set_xlim(time_lin_1[0], time_lin_1[-1])
ax1.set_ylim(20., 80.)
ax2.set_xlim(time_lin_1[0], time_lin_1[-1])
ax1.set_ylabel('Force (N)', fontsize=56)
ax2.set_ylabel(r'$\Delta F$ Estimate (N)', fontsize=56)
ax2.set_xlabel('Time (s)', fontsize=56)

ax1.tick_params(axis = 'y', labelsize=48)
ax2.tick_params(axis = 'x', labelsize=48)
ax2.tick_params(axis = 'y', labelsize=48)
ax1.tick_params(labelbottom=False)  


if PART == 1:
    line_f_r1, = ax1.plot(time_lin_1[0:1], force_1[0:1], animated=True, color=color_list[1], linewidth=6, label=label1, alpha=0.8)
    line_df_r1, = ax2.plot(time_lin_1[0:1], np.zeros(time_lin_1[0:1].shape), animated=True, color=color_list[1], linewidth=6, label=label1, alpha=0.8)
    line = [line_f_r1, line_df_r1]
    
if PART == 2:
    line_f_r1, = ax1.plot(time_lin_1[:], force_1[:], animated=True, color=color_list[1], linewidth=6, label=label1, alpha=0.8)
    line_f_r2, = ax1.plot(time_lin_2[0:1], force_2[0:1], animated=True, color=color_list[2], linewidth=6, label=label2, alpha=0.8)


    line_df_r1, = ax2.plot(time_lin_2[:], np.zeros(time_lin_2.shape), animated=True, color=color_list[1], linewidth=6, label=label1, alpha=0.8)
    line_df_r2, = ax2.plot(time_lin_2[0:1], delta_F[0:1], animated=True, color=color_list[2], linewidth=6, label=label2, alpha=0.8)
    line = [line_f_r1, line_df_r1, line_f_r2, line_df_r2]

# Task phases
PHASE_TIME = 6283
ax1.axvline(PHASE_TIME/1000., color='k', linewidth=4, linestyle='-', alpha=1.)
ax1.axvline(2*PHASE_TIME/1000., color='k', linewidth=4, linestyle='-', alpha=1.)
ax2.axvline(PHASE_TIME/1000., color='k', linewidth=4, linestyle='-', alpha=1.)
ax2.axvline(2*PHASE_TIME/1000., color='k', linewidth=4, linestyle='-', alpha=1.)
fig.text(0.22, 0.90, 'Slow', transform=fig.transFigure, fontdict={'size':70})
fig.text(0.5, 0.90, 'Medium',transform=fig.transFigure, fontdict={'size':70})
fig.text(0.82, 0.90, 'Fast',transform=fig.transFigure, fontdict={'size':70})

if PART == 1:
    fig.text(0.48, 0.3, 'No estimation',transform=fig.transFigure, fontdict={'size':70})

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
        mask1 = time_lin_1 < t
        line[0].set_data(time_lin_1[mask1], force_1[mask1])
        line[1].set_data(time_lin_1[mask1], np.zeros(time_lin_1[mask1].shape))
    if PART == 2:
        mask2 = time_lin_2 < t
        line[2].set_data(time_lin_2[mask2], force_2[mask2])
        line[3].set_data(time_lin_2[mask2], delta_F[mask2])


    return line

# Create FuncAnimation object and plt.show() to show the updated animation


t0 = time.time()

time_lin = np.linspace(0, T, N_FRAMES)
ani = FuncAnimation(fig, animate, frames=time_lin, repeat=False, interval = SKIP, init_func = init, blit=True)
folder = data_path #'/home/skleff/Desktop/delta_f_real_exp/video/'

ani.save(folder + 'animation_FL_vs_DF_perturbation_' + str(PART) + '.mp4')


print("COMPUTE TIME = ", time.time() - t0)
# plt.show()



