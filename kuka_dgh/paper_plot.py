
import matplotlib
matplotlib.use('GTK3Agg') 


from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
# matplotlib.rcParams["pdf.fonttype"] = 42
# matplotlib.rcParams["ps.fonttype"] = 42
import numpy as np

import time
from matplotlib import pyplot as plt
import numpy as np

# plt.ion()

# fig = plt.figure()
# ax1 = fig.add_subplot(2, 1, 1)
# ax2 = fig.add_subplot(2, 1, 2)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(19.2,10.8))   
fig.canvas.draw()   # note that the first draw comes before setting data 
# cache the background
axbackground = fig.canvas.copy_from_bbox(ax1.bbox)
ax2background = fig.canvas.copy_from_bbox(ax2.bbox)
# plt.show(block=False)


# ax1.legend(framealpha=0.95, fontsize=26)     
        
ax1.grid(True) 
ax2.grid(True) 
ax1.set_xlim(0, 1)
ax2.set_xlim(0, 1)
ax1.set_ylim(-1.1, 1.1)
ax2.set_ylim(-1.1, 1.1)
ax1.set_ylabel('Force (N)', fontsize=26)
ax2.set_ylabel('Position error (m)', fontsize=26)
ax2.set_xlabel('Time (s)', fontsize=26)
ax1.tick_params(axis = 'y', labelsize=22)
ax2.tick_params(axis = 'x', labelsize=22)
ax2.tick_params(axis = 'y', labelsize=22)
ax1.tick_params(labelbottom=False)  
# fig0.tight_layout()


line1, = ax1.plot([], [], animated=True, label="hello")
line2, = ax2.plot([], [], animated=True, label="hello")
line = [line1, line2]

N = 100

x = np.linspace(0, 1, N)
y1 = np.sin(x)
y2 = np.cos(x)

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

    line[0].set_data(x[:t], y1[:t])
    line[1].set_data(x[:t], y2[:t])

    return line

# Create FuncAnimation object and plt.show() to show the updated animation

import time

t0 = time.time()

Frames = [i for i in range(0, N)]
ani = FuncAnimation(fig, animate, frames=Frames, repeat=False, interval = 10, init_func = init, blit=True)
ani.save('animation.mp4')


print("COMPUTE TIME = ", time.time() - t0)
# plt.show()


assert False





 







import sys
sys.path.append("../../")
from force_feedback_dgh.demos.utils.plot_utils import SimpleDataPlotter
from mim_data_utils import DataReader
from core_mpc.pin_utils import *
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt 
from core_mpc import path_utils
from core_mpc.pin_utils import get_p_, get_v_, get_rpy_
from core_mpc import analysis_utils 

from robot_properties_kuka.config import IiwaConfig, IiwaReducedConfig


from robot_properties_kuka.config import IiwaConfig
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
N_START = 7000 

N_START = int(config['T_CIRCLE'] * config['simu_freq']) 
print("N_start = ", N_START)

data_path = '/home/skleff/Desktop/delta_f_real_exp/video/'


print("Load data 1...")
r1 = DataReader(data_path+'config_REAL_2023-09-19T12:17:31.957715_test.mds') 
print("Load data 2...")
r2 = r1


label1 = 'Default'
label2 = r'FL + $\Delta F$ (PM)'


N = min(r1.data['tau'].shape[0], r2.data['tau'].shape[0])

 
print("N_start = ", N_START)




force_3d_1 = np.zeros((N-N_START, 3))
force_3d_2 = np.zeros((N-N_START, 3))
force_3d_3 = np.zeros((N-N_START, 3))
force_3d_4 = np.zeros((N-N_START, 3))



if(FILTER > 0):
    force_3d_1 = analysis_utils.moving_average_filter(r1.data['contact_force_3d_measured'][N_START:N].copy(), FILTER)
    force_3d_2 = analysis_utils.moving_average_filter(r2.data['contact_force_3d_measured'][N_START:N].copy(), FILTER) 
else:
    force_3d_1 = r1.data['contact_force_3d_measured'][N_START:N]
    force_3d_2 = r2.data['contact_force_3d_measured'][N_START:N] 

from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
 
color_list = ['b', 'g', 'r', 'y']


def compute_pos_error_traj(r):
    p_mea = get_p_(r.data['joint_positions'][N_START:,controlled_joint_ids], pinrobot.model, pinrobot.model.getFrameId('contact'))
    return np.sqrt((p_mea[:, 0] - r.data['target_position_x'][N_START:,0])**2 + (p_mea[:, 1] - r.data['target_position_y'][N_START:,1])**2)


pos_error_1 = np.zeros((N-N_START, 1))
pos_error_2 = np.zeros((N-N_START, 1))

pos_error_1 = compute_pos_error_traj(r1)
pos_error_2 = compute_pos_error_traj(r2)


target_force_3d = np.zeros((N, 3))
Tc = int(config['T_CONTACT'] * config['simu_freq'])
target_force_3d[Tc:, :3] = np.asarray(config['frameForceRef'])[:3]
N_ramp = int((config['T_RAMP'] - config['T_CONTACT']) * config['simu_freq'])           
FZ_MIN = 5. 
FZ_MAX = 50.
FX_MAX = config['frameForceRef'][0]
target_force_3d[Tc:Tc+N_ramp, 2] = [FZ_MIN + (FZ_MAX - FZ_MIN)*i/N_ramp for i in range(N_ramp)]
target_force_3d[Tc+N_ramp:, 2] = FZ_MAX
time_lin = np.linspace(0, (N-N_START) / config['simu_freq'], (N-N_START))




#Initiate figure and ax objects

fig0, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 18))


ax1.plot(time_lin, target_force_3d[N_START:N,2], color='k', linewidth=4, linestyle='--', label='Reference', alpha=0.5) 
ax1.grid(True) 
ax2.grid(True) 
ax2.set_ylim(0., 0.02)
ax1.set_xlim(time_lin[0], time_lin[-1])
ax1.set_ylim(0., 100.)
ax2.set_xlim(time_lin[0], time_lin[-1])
ax1.legend(framealpha=0.95, fontsize=26)     
ax1.set_ylabel('Force (N)', fontsize=26)
ax2.set_ylabel('Position error (m)', fontsize=26)
ax2.set_xlabel('Time (s)', fontsize=26)
ax1.tick_params(axis = 'y', labelsize=22)
ax2.tick_params(axis = 'x', labelsize=22)
ax2.tick_params(axis = 'y', labelsize=22)
ax1.tick_params(labelbottom=False)  
fig0.tight_layout()


x, y = [], []
line1, = ax1.plot([], [], 'bo')
# line,  = ax1.plot(time_lin, force_3d_1[:, 2], color=color_list[0], linewidth=4, label=label1, alpha=0.8)
# line,  = ax1.plot(time_lin, force_3d_2, color=color_list[1], linewidth=4, label=label2, alpha=0.8)
# line,  = ax2.plot(time_lin, pos_error_1, color=color_list[0], linewidth=4, label=label1, alpha=0.8)
# line,  = ax2.plot(time_lin, pos_error_2, color=color_list[1], linewidth=4, label=label2, alpha=0.8)


# ax1.plot(time_lin[:, 2], force_3d_1[:,2], color=color_list[0], linewidth=4, label=label1, alpha=0.8)
# ax1.plot(time_lin[:, 2], force_3d_2[:,2], color=color_list[1], linewidth=4, label=label2, alpha=0.8)
# ax2.plot(time_lin[:, 2], pos_error_1[:2], color=color_list[0], linewidth=4, label=label1, alpha=0.8)
# ax2.plot(time_lin[:, 2], pos_error_2[:2], color=color_list[1], linewidth=4, label=label2, alpha=0.8)









def init():
    """
    This init function defines the initial plot parameter
    """
    # Set initial parameter for the plot
    return line1,

def animate(frame):
    """
    This function will be called periodically by FuncAnimation. Frame parameter will be passed on each call as a counter. 
    """
    # Append data to x and y data list

    t = int(frame*1000)
    
    line1.set_data(time_lin[:t:100], force_3d_1[:t:100, 2])
    return line1,

# Create FuncAnimation object and plt.show() to show the updated animation
ani = FuncAnimation(fig0, animate, frames = time_lin, interval = 100, init_func = init)
ani.save('animation.mp4')

plt.show()


assert False


