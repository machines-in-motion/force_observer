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
r1 = DataReader(data_path+'config_REAL_2023-09-20T10:53:42.357246_DEFAULT.mds') 
print("Load data 2...")
# r2 = r1 
r2 = DataReader(data_path+'config_REAL_2023-09-20T11:26:26.197428_FL.mds') 
print("Load data 3...")
# r3 = r1 
r3 = DataReader(data_path+'config_REAL_2023-09-20T10:51:44.473581_FL_DF_PM.mds') 


label1 = 'Default'
label2 = 'FL'
label3 = r'FL + $\Delta F$ (PM)'


N = min(r1.data['tau'].shape[0], r2.data['tau'].shape[0])
N = 25000
 




force_3d_1 = np.zeros((N-N_START, 3))
force_3d_2 = np.zeros((N-N_START, 3))
force_3d_3 = np.zeros((N-N_START, 3))


if(FILTER > 0):
    force_3d_1 = analysis_utils.moving_average_filter(r1.data['contact_force_3d_measured'][N_START:N].copy(), FILTER)
    force_3d_2 = analysis_utils.moving_average_filter(r2.data['contact_force_3d_measured'][N_START:N].copy(), FILTER) 
    force_3d_3 = analysis_utils.moving_average_filter(r3.data['contact_force_3d_measured'][N_START:N].copy(), FILTER) 
else:
    force_3d_1 = r1.data['contact_force_3d_measured'][N_START:N]
    force_3d_2 = r2.data['contact_force_3d_measured'][N_START:N] 
    force_3d_3 = r3.data['contact_force_3d_measured'][N_START:N] 



def compute_pos_error_traj(r):
    p_mea = get_p_(r.data['joint_positions'][N_START:N,controlled_joint_ids], pinrobot.model, pinrobot.model.getFrameId('contact'))
    return np.sqrt((p_mea[:, 0] - r.data['target_position_x'][N_START:N,0])**2 + (p_mea[:, 1] - r.data['target_position_y'][N_START:N,1])**2)



pos_error_1 = compute_pos_error_traj(r1)
pos_error_2 = compute_pos_error_traj(r2)
pos_error_3 = compute_pos_error_traj(r3)
    
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



color_list = ['b', 'g', 'r', 'y']







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



fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(30, 18))

fig.canvas.draw() 

ax1.plot(time_lin, target_force_3d[N_START:N,2], color='k', linewidth=4, linestyle='--', label='Reference', alpha=0.5) 
ax1.grid(True) 
ax2.grid(True) 
ax2.set_ylim(0., 0.03)
ax1.set_xlim(time_lin[0], time_lin[-1])
ax1.set_ylim(25., 90.)
ax2.set_xlim(time_lin[0], time_lin[-1])
# ax1.legend(framealpha=0.95, fontsize=28)     
ax1.set_ylabel('Force (N)', fontsize=30)
ax2.set_ylabel('Position error (m)', fontsize=30)
ax2.set_xlabel('Time (s)', fontsize=30)
ax1.tick_params(axis = 'y', labelsize=26)
ax2.tick_params(axis = 'x', labelsize=26)
ax2.tick_params(axis = 'y', labelsize=26)
ax1.tick_params(labelbottom=False)  


line1a, = ax1.plot(time_lin[0:1], force_3d_1[0:1, 2], animated=True, color=color_list[0], linewidth=4, label=label1, alpha=0.8)
line1b, = ax1.plot(time_lin[0:1], force_3d_2[0:1, 2], animated=True, color=color_list[1], linewidth=4, label=label2, alpha=0.8)
line1c, = ax1.plot(time_lin[0:1], force_3d_3[0:1, 2], animated=True, color=color_list[2], linewidth=4, label=label3, alpha=0.8)


line2a, = ax2.plot(time_lin[0:1], pos_error_1[0:1], animated=True, color=color_list[0], linewidth=4, label=label1, alpha=0.8)
line2b, = ax2.plot(time_lin[0:1], pos_error_2[0:1], animated=True, color=color_list[1], linewidth=4, label=label2, alpha=0.8)
line2c, = ax2.plot(time_lin[0:1], pos_error_3[0:1], animated=True, color=color_list[2], linewidth=4, label=label3, alpha=0.8)

# Task phases
PHASE_TIME = 6283
ax1.axvline(PHASE_TIME/1000., color='k', linewidth=2, linestyle='-', alpha=1.)
ax1.axvline(2*PHASE_TIME/1000., color='k', linewidth=2, linestyle='-', alpha=1.)
ax2.axvline(PHASE_TIME/1000., color='k', linewidth=2, linestyle='-', alpha=1.)
ax2.axvline(2*PHASE_TIME/1000., color='k', linewidth=2, linestyle='-', alpha=1.)
fig.text(0.22, 0.92, 'Slow', transform=fig.transFigure, fontdict={'size':35})
fig.text(0.5, 0.92, 'Medium',transform=fig.transFigure, fontdict={'size':35})
fig.text(0.75, 0.92, 'Fast',transform=fig.transFigure, fontdict={'size':35})

line = [line1a, line1b, line1c, line2a, line2b, line2c]

ax1.legend(loc="upper left", framealpha=0.95, fontsize=28) 

fig.align_ylabels()
# fig.tight_layout()


T = int((N-N_START)/1000)
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

    line[0].set_data(time_lin[0:t:SKIP], force_3d_1[0:t:SKIP, 2])
    line[1].set_data(time_lin[0:t:SKIP], force_3d_2[0:t:SKIP, 2])
    line[2].set_data(time_lin[0:t:SKIP], force_3d_3[0:t:SKIP, 2])


    line[3].set_data(time_lin[0:t:SKIP], pos_error_1[0:t:SKIP])
    line[4].set_data(time_lin[0:t:SKIP], pos_error_2[0:t:SKIP])
    line[5].set_data(time_lin[0:t:SKIP], pos_error_3[0:t:SKIP])


    return line

# Create FuncAnimation object and plt.show() to show the updated animation

import time

t0 = time.time()

Frames = [i * SKIP for i in range(0, N_FRAMES)]
ani = FuncAnimation(fig, animate, frames=Frames, repeat=False, interval = SKIP, init_func = init, blit=True)
ani.save('animation_delault_vs_FL_vs_FL_DF.mp4')


print("COMPUTE TIME = ", time.time() - t0)
# plt.show()



