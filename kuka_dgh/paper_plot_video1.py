import sys
sys.path.append("../../")
from force_feedback_dgh.demos.utils.plot_utils import SimpleDataPlotter
from mim_data_utils import DataReader
from core_mpc.pin_utils import *
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt 
from core_mpc import path_utils
from core_mpc.pin_utils import get_p_
from core_mpc import analysis_utils 

from robot_properties_kuka.config import IiwaConfig, IiwaReducedConfig

import matplotlib
matplotlib.use('GTK3Agg') 
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import time



pinrobot = IiwaConfig.buildRobotWrapper()
model = pinrobot.model
data = model.createData()
frameId = model.getFrameId('contact')
nq = model.nq ; nv = model.nv ; nc = 3
# Overwrite effort limit as in DGM
model.effortLimit = np.array([100, 100, 50, 50, 20, 10, 10])



# Load robot model
CONTROLLED_JOINTS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
QREF              = np.zeros(7)
pinrobot = IiwaReducedConfig.buildRobotWrapper(controlled_joints=CONTROLLED_JOINTS, qref=QREF)
full_robot = IiwaConfig.buildRobotWrapper()
full_model = full_robot.model
model = pinrobot.model
data = model.createData()
frameId = model.getFrameId('contact')
nq = model.nq ; nv = model.nv ; nc = 1
# Overwrite effort limit as in DGM
model.effortLimit = np.array([100, 100, 50, 50, 20, 10, 10])
controlled_joint_ids = [full_model.joints[full_model.getJointId(joint_name)].idx_q for joint_name in CONTROLLED_JOINTS]
print(controlled_joint_ids)
# Load config file
CONFIG_NAME = 'config'
# CONFIG_NAME = 'config36d'

CONFIG_PATH = CONFIG_NAME+".yml"
config      = path_utils.load_yaml_file(CONFIG_PATH)


# Load data 
SAVE = False

# Create data Plottger
s = SimpleDataPlotter()

FILTER = 400


data_path = '/home/skleff/Desktop/delta_f_real_exp/video/'


print("Load data 1...")
r1 = DataReader(data_path+'config_REAL_2023-09-21T11:56:08.250995_DEFAULT.mds') 
print("Load data 2...")
# r2 = r1 
r2 = DataReader(data_path+'config_REAL_2023-09-21T11:54:57.055958_FL.mds') 
print("Load data 3...")
# r3 = r1 
r3 = DataReader(data_path+'config_REAL_2023-09-21T11:53:34.390174_FL_DF_PM.mds') 


label1 = 'Default'
label2 = 'FL'
label3 = r'FL + $\Delta F$'


N_TOT = min(r1.data['tau'].shape[0], r2.data['tau'].shape[0])
N_TOT = min(N_TOT, r3.data['tau'].shape[0])

T = 6283 * 3 / config['plan_freq']
 


time_lin1 = r1.data['time']
time_lin2 = r2.data['time']
time_lin3 = r3.data['time']

N_START_1 = np.sum(time_lin1 < config['T_CIRCLE'])
N_START_2 = np.sum(time_lin2 < config['T_CIRCLE'])
N_START_3 = np.sum(time_lin3 < config['T_CIRCLE'])

N_END_1 = np.sum(time_lin1 <= T + config['T_CIRCLE'])
N_END_2 = np.sum(time_lin2 <= T + config['T_CIRCLE'])
N_END_3 = np.sum(time_lin3 <= T + config['T_CIRCLE'])



if(FILTER > 0):
    print("FILTERING")
    force_1 = analysis_utils.moving_average_filter(r1.data['contact_force_3d_measured'][N_START_1:N_END_1, 2:3].copy(), FILTER)
    force_2 = analysis_utils.moving_average_filter(r2.data['contact_force_3d_measured'][N_START_2:N_END_2, 2:3].copy(), FILTER) 
    force_3 = analysis_utils.moving_average_filter(r3.data['contact_force_3d_measured'][N_START_3:N_END_3, 2:3].copy(), FILTER) 
else:
    force_1 = r1.data['contact_force_3d_measured'][N_START_1:N_END_1, 2:3]
    force_2 = r2.data['contact_force_3d_measured'][N_START_2:N_END_2, 2:3] 
    force_3 = r3.data['contact_force_3d_measured'][N_START_3:N_END_3, 2:3] 


def compute_pos_error_traj(r):
    p_mea = get_p_(r.data['joint_positions'][:N_TOT,controlled_joint_ids], pinrobot.model, pinrobot.model.getFrameId('contact'))
    return (np.sqrt((p_mea[:, 0] - r.data['target_position_x'][:N_TOT,0])**2 + (p_mea[:, 1] - r.data['target_position_y'][:N_TOT,1])**2)).reshape((-1, 1))



pos_error_1 = analysis_utils.moving_average_filter(compute_pos_error_traj(r1)[N_START_1:N_END_1].copy(), 100)
pos_error_2 = analysis_utils.moving_average_filter(compute_pos_error_traj(r2)[N_START_2:N_END_2].copy(), 100)
pos_error_3 = analysis_utils.moving_average_filter(compute_pos_error_traj(r3)[N_START_3:N_END_3].copy(), 100)
    


SPLIT = 10
time_lin_1 = time_lin1[N_START_1:N_END_1:SPLIT] - config['T_CIRCLE']
time_lin_2 = time_lin2[N_START_2:N_END_2:SPLIT] - config['T_CIRCLE']
time_lin_3 = time_lin3[N_START_3:N_END_3:SPLIT] - config['T_CIRCLE']



force_1 = force_1[::SPLIT]
force_2 = force_2[::SPLIT]
force_3 = force_3[::SPLIT]

pos_error_1 = pos_error_1[::SPLIT]
pos_error_2 = pos_error_2[::SPLIT]
pos_error_3 = pos_error_3[::SPLIT]



target_force = np.zeros(time_lin_1.shape)
target_force[:] = config['frameForceRef'][2]



color_list = ['b', 'g', 'r', 'y']


print("PLOTTING")


fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(68, 16))

fig.canvas.draw() 

ax1.plot(time_lin_1, target_force, color='k', linewidth=6, linestyle='--', label='Reference', alpha=0.5) 
ax1.grid(linewidth=1)
ax2.grid(linewidth=1) 
ax2.set_ylim(0., 0.03)
ax1.set_xlim(time_lin_1[0], time_lin_1[-1])
ax1.set_ylim(25., 90.)
ax2.set_xlim(time_lin_1[0], time_lin_1[-1])
ax1.set_ylabel('Force (N)', fontsize=56)
ax2.set_ylabel('Position error (m)', fontsize=56)
ax2.set_xlabel('Time (s)', fontsize=56)

ax1.tick_params(axis = 'y', labelsize=48)
ax2.tick_params(axis = 'x', labelsize=48)
ax2.tick_params(axis = 'y', labelsize=48)
ax1.tick_params(labelbottom=False)  
ax1.set_yticks([30, 50, 70, 90])

line1a, = ax1.plot(time_lin_1[0:1], force_1[0:1], animated=True, color=color_list[0], linewidth=6, label=label1, alpha=0.8)
line1b, = ax1.plot(time_lin_2[0:1], force_2[0:1], animated=True, color=color_list[1], linewidth=6, label=label2, alpha=0.8)
line1c, = ax1.plot(time_lin_3[0:1], force_3[0:1], animated=True, color=color_list[2], linewidth=6, label=label3, alpha=0.8)


line2a, = ax2.plot(time_lin_1[0:1], pos_error_1[0:1], animated=True, color=color_list[0], linewidth=6, label=label1, alpha=0.8)
line2b, = ax2.plot(time_lin_2[0:1], pos_error_2[0:1], animated=True, color=color_list[1], linewidth=6, label=label2, alpha=0.8)
line2c, = ax2.plot(time_lin_3[0:1], pos_error_3[0:1], animated=True, color=color_list[2], linewidth=6, label=label3, alpha=0.8)

# Task phases
PHASE_TIME = 6283
ax1.axvline(PHASE_TIME/1000., color='k', linewidth=4, linestyle='-', alpha=1.)
ax1.axvline(2*PHASE_TIME/1000., color='k', linewidth=4, linestyle='-', alpha=1.)
ax2.axvline(PHASE_TIME/1000., color='k', linewidth=4, linestyle='-', alpha=1.)
ax2.axvline(2*PHASE_TIME/1000., color='k', linewidth=4, linestyle='-', alpha=1.)
fig.text(0.22, 0.90, 'Slow', transform=fig.transFigure, fontdict={'size':70})
fig.text(0.5, 0.90, 'Medium',transform=fig.transFigure, fontdict={'size':70})
fig.text(0.82, 0.90, 'Fast',transform=fig.transFigure, fontdict={'size':70})

line = [line1a, line1b, line1c, line2a, line2b, line2c]

ax1.legend(loc="upper left", framealpha=0.95, fontsize=40) 

fig.align_ylabels()
plt.tight_layout(pad=1)


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
    
    mask1 = time_lin_1 < t
    line[0].set_data(time_lin_1[mask1], force_1[mask1])
    line[3].set_data(time_lin_1[mask1], pos_error_1[mask1])

    mask2 = time_lin_2 < t
    line[4].set_data(time_lin_2[mask2], pos_error_2[mask2])
    line[1].set_data(time_lin_2[mask2], force_2[mask2])

    mask3 = time_lin_3 < t
    line[2].set_data(time_lin_3[mask3], force_3[mask3])
    line[5].set_data(time_lin_3[mask3], pos_error_3[mask3])

    return line

# Create FuncAnimation object and plt.show() to show the updated animation

import time

t0 = time.time()


time_lin = np.linspace(0, T, N_FRAMES)
ani = FuncAnimation(fig, animate, frames=time_lin, repeat=False, interval = SKIP, init_func = init, blit=True)
folder = '/home/skleff/Desktop/delta_f_real_exp/video/'
ani.save(folder + 'animation_delault_vs_FL_vs_FL_DF.mp4', fps=PPS)


print("COMPUTE TIME = ", time.time() - t0)
# plt.show()



