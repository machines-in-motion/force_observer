import sys
sys.path.append("../../")
from force_feedback_dgh.demos.utils.plot_utils import SimpleDataPlotter
from mim_data_utils import DataReader
from core_mpc.pin_utils import *
import pinocchio as pin
from core_mpc import path_utils
from core_mpc.pin_utils import get_p_, get_v_, get_rpy_
from core_mpc import analysis_utils 


import matplotlib
matplotlib.use('GTK3Agg') 
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
import numpy as np

import time

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
CONFIG_NAME = 'config36d'

CONFIG_PATH = CONFIG_NAME+".yml"
config      = path_utils.load_yaml_file(CONFIG_PATH)


# Load data 

# Create data Plottger
s = SimpleDataPlotter()

FILTER = 400


data_path =  '/home/skleff/Desktop/delta_f_real_exp/video/'
    
label1 = r'Force Integral'
label2 = r'$\Delta F$ (PM)'


print("Load data 1...")
r1 = DataReader(data_path+'config36d_REAL_2023-09-21T18:52:55.022903_energy_integral.mds')
print("Load data 2...")
r2 = DataReader(data_path+'config36d_REAL_2023-09-21T18:50:58.342199_energy_df.mds')


N_TOT = min(r1.data['tau'].shape[0], r2.data['tau'].shape[0])

T = 20000 / config['plan_freq']

T_START = 7 + 2

time_lin1 = r1.data['time'] / config['simu_freq']
time_lin2 = r2.data['time'] / config['simu_freq']

N_START_1 = np.sum(time_lin1 < T_START)
N_START_2 = np.sum(time_lin2 < T_START)

N_END_1 = np.sum(time_lin1 <= T + T_START)
N_END_2 = np.sum(time_lin2 <= T + T_START)


if(FILTER > 0):
    print("FILTERING")
    r1.data['contact_force_3d_measured'] = analysis_utils.moving_average_filter(r1.data['contact_force_3d_measured'].copy(), FILTER)
    r2.data['contact_force_3d_measured'] = analysis_utils.moving_average_filter(r2.data['contact_force_3d_measured'].copy(), FILTER) 

    
# Compute energy
  
SPLIT = 10

def compute_all_cost_terms(r, N_START, N_END):   
    cost_list = []
    force_cost_list = []
    state_cost_list = []
    tau_cost_list = []
    rotation_cost_list = []
    for t in range(0, N_END - N_START, SPLIT):
        index = N_START  + t
        f = r.data['contact_force_3d_measured'][index][:3]   
        q = r.data['joint_positions'][index,controlled_joint_ids]
        v = r.data['joint_velocities'][index,controlled_joint_ids]
        x = np.concatenate([q, v])

        tau = r.data['tau'][index, controlled_joint_ids] + r.data['tau_gravity'][index,controlled_joint_ids]
        rotation = pin.utils.rpyToMatrix(get_rpy_(r.data['joint_positions'][index,controlled_joint_ids], pinrobot.model, pinrobot.model.getFrameId('contact')))
        f_ref = config['frameForceRef'][:3]
        xref = np.array([0., 1.0471975511965976, r.data['target_joint'][index, 0], -1.1344640137963142, 0.2,  0.7853981633974483, 0.,0.,0.,0.,0.,0.])
        tau_ref = np.zeros(tau.shape)
        rotation_ref = pin.utils.rpyToMatrix(np.pi, 0, np.pi)


        force_cost = 0.5 * config['frameForceWeight'] * (f - f_ref).T @ np.diag(config['frameForceWeights'][:3])@ (f - f_ref)
        force_cost_list.append(force_cost)


        state_cost = 0.5 * config['stateRegWeight'] * (x - xref).T @ np.diag(config['stateRegWeights'])@ (x - xref)
        state_cost_list.append(state_cost)


        tau_cost =  0.5 * 0.0008 * (tau - tau_ref).T @ (tau - tau_ref)
        tau_cost_list.append(tau_cost)


        rot_residual = pin.log3(rotation_ref.T @ rotation) 
        rotation_cost = 0.5 * config['frameRotationWeight'] * rot_residual.T @ np.diag(config['frameRotationWeights']) @ rot_residual
        rotation_cost_list.append(rotation_cost)


        cost_list.append(force_cost + state_cost + tau_cost + rotation_cost)
    f_cost = np.array(force_cost_list).reshape(-1, 1)
    s_cost = np.array(state_cost_list).reshape(-1, 1)
    t_cost = np.array(tau_cost_list).reshape(-1, 1)
    r_cost = np.array(rotation_cost_list).reshape(-1, 1)
    c_cost = np.array(cost_list).reshape(-1, 1)
    return f_cost, s_cost, t_cost, r_cost, c_cost

force_cost_list_r1, state_cost_list_r1, tau_cost_list_r1, rotation_cost_list_r1, cost_list_r1 = compute_all_cost_terms(r1, N_START_1, N_END_1)
force_cost_list_r2, state_cost_list_r2, tau_cost_list_r2, rotation_cost_list_r2, cost_list_r2 = compute_all_cost_terms(r2, N_START_2, N_END_2)



time_lin_1 = time_lin1[N_START_1:N_END_1:SPLIT] - T_START
time_lin_2 = time_lin2[N_START_2:N_END_2:SPLIT] - T_START


color_list = ['b', 'r']


ANIMATION = True
LINEWIDTH = 4
ALPHA = 0.8
 

print("PLOTTING")

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex='col', figsize=(55, 22))
 

ax1.grid(linewidth=1) 
ax2.grid(linewidth=1) 
ax3.grid(linewidth=1) 
ax4.grid(linewidth=1) 

ax1.set_xlim(time_lin_1[0], time_lin_1[-1])
ax2.set_xlim(time_lin_1[0], time_lin_1[-1])
ax3.set_xlim(time_lin_1[0], time_lin_1[-1])
ax4.set_xlim(time_lin_1[0], time_lin_1[-1])

ax1.set_ylim(0., 0.7)
ax2.set_ylim(0., 0.02)
ax3.set_ylim(0., 1.1)
ax4.set_ylim(0., 1.6)
   
ax1.set_ylabel('Energy cost ', fontsize=45)
ax2.set_ylabel('Force cost ' , fontsize=45)
ax3.set_ylabel('Other terms ', fontsize=45)
ax4.set_ylabel('Total cost ' , fontsize=45)

ax4.set_xlabel('Time (s)', fontsize=45)

ax1.tick_params(axis = 'y', labelsize=38)
ax2.tick_params(axis = 'y', labelsize=38)
ax3.tick_params(axis = 'y', labelsize=38)
ax4.tick_params(axis = 'y', labelsize=38)

ax4.tick_params(axis = 'x', labelsize=38)

ax1.tick_params(labelbottom=False)  
ax2.tick_params(labelbottom=False)  
ax3.tick_params(labelbottom=False)  


if ANIMATION:
    fig.canvas.draw() 

    line1_r1, = ax1.plot(time_lin_1[0:1], tau_cost_list_r1[0:1], animated=True, color=color_list[0], linewidth=LINEWIDTH, label=label1, alpha=ALPHA)
    line2_r1, = ax2.plot(time_lin_1[0:1], force_cost_list_r1[0:1], animated=True, color=color_list[0], linewidth=LINEWIDTH, label=label1, alpha=ALPHA)
    line3_r1, = ax3.plot(time_lin_1[0:1], state_cost_list_r1[0:1] + rotation_cost_list_r1[0:1], animated=True, color=color_list[0], linewidth=LINEWIDTH, label=label1, alpha=ALPHA)
    line4_r1, = ax4.plot(time_lin_1[0:1], cost_list_r1[0:1] , animated=True, color=color_list[0], linewidth=LINEWIDTH, label=label1, alpha=ALPHA)

    line1_r2, = ax1.plot(time_lin_2[0:1], tau_cost_list_r2[0:1], animated=True, color=color_list[1], linewidth=LINEWIDTH, label=label2, alpha=ALPHA)
    line2_r2, = ax2.plot(time_lin_2[0:1], force_cost_list_r2[0:1], animated=True, color=color_list[1], linewidth=LINEWIDTH, label=label2, alpha=ALPHA)
    line3_r2, = ax3.plot(time_lin_2[0:1], state_cost_list_r2[0:1] + rotation_cost_list_r2[0:1], animated=True, color=color_list[1], linewidth=LINEWIDTH, label=label2, alpha=ALPHA)
    line4_r2, = ax4.plot(time_lin_2[0:1], cost_list_r2[0:1], animated=True, color=color_list[1], linewidth=LINEWIDTH, label=label2, alpha=ALPHA)
   
else:
    line1_r1, = ax1.plot(time_lin_1, tau_cost_list_r1, animated=False, color=color_list[0], linewidth=LINEWIDTH, label=label1, alpha=ALPHA)
    line2_r1, = ax2.plot(time_lin_1, force_cost_list_r1, animated=False, color=color_list[0], linewidth=LINEWIDTH, label=label1, alpha=ALPHA)
    line3_r1, = ax3.plot(time_lin_1, state_cost_list_r1 + rotation_cost_list_r1, animated=False, color=color_list[0], linewidth=LINEWIDTH, label=label1, alpha=ALPHA)
    line4_r1, = ax4.plot(time_lin_1, cost_list_r1 , animated=False, color=color_list[0], linewidth=LINEWIDTH, label=label1, alpha=ALPHA)

    line1_r2, = ax1.plot(time_lin_2, tau_cost_list_r2, animated=False, color=color_list[1], linewidth=LINEWIDTH, label=label2, alpha=ALPHA)
    line2_r2, = ax2.plot(time_lin_2, force_cost_list_r2, animated=False, color=color_list[1], linewidth=LINEWIDTH, label=label2, alpha=ALPHA)
    line3_r2, = ax3.plot(time_lin_2, state_cost_list_r2 + rotation_cost_list_r2, animated=False, color=color_list[1], linewidth=LINEWIDTH, label=label2, alpha=ALPHA)
    line4_r2, = ax4.plot(time_lin_2, cost_list_r2 , animated=False, color=color_list[1], linewidth=LINEWIDTH, label=label2, alpha=ALPHA)



ax1.legend(loc="upper right", framealpha=0.95, fontsize=30) 
fig.align_ylabels()
plt.tight_layout(pad=6)


if ANIMATION:

    line = [line1_r1, line2_r1, line3_r1, line4_r1, line1_r2, line2_r2, line3_r2, line4_r2]

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
        # import pdb; pdb.set_trace()
        line[0].set_data(time_lin_1[mask1], tau_cost_list_r1[mask1])
        line[1].set_data(time_lin_1[mask1], force_cost_list_r1[mask1])
        line[2].set_data(time_lin_1[mask1], state_cost_list_r1[mask1]+rotation_cost_list_r1[mask1])
        line[3].set_data(time_lin_1[mask1], cost_list_r1[mask1])
        
        mask2 = time_lin_2 < t
        line[4].set_data(time_lin_2[mask2], tau_cost_list_r2[mask2])
        line[5].set_data(time_lin_2[mask2], force_cost_list_r2[mask2])
        line[6].set_data(time_lin_2[mask2], state_cost_list_r2[mask2]+rotation_cost_list_r2[mask2])
        line[7].set_data(time_lin_2[mask2], cost_list_r2[mask2])

        
        # fig.text(0.5, 0.92, str(np.mean(round(cost_list_r2[0:t:SKIP], 2))), transform=fig.transFigure, fontdict={'size':35})
        
        return line


    t0 = time.time()

    time_lin = np.linspace(0, T, N_FRAMES)
    ani = FuncAnimation(fig, animate, frames=time_lin, repeat=False, interval = SKIP, init_func = init, blit=True)
    folder = '/home/skleff/Desktop/delta_f_real_exp/video/'
    ani.save(folder + 'animation_energy.mp4')


    print("COMPUTE TIME = ", time.time() - t0)


else:
    plt.show()