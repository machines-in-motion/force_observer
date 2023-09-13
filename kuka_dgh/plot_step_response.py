import sys
sys.path.append("/home/skleff/ws/workspace/src/")
from force_feedback_dgh.demos.utils.plot_utils import SimpleDataPlotter
from mim_data_utils import DataReader
from core_mpc.pin_utils import *
from core_mpc import path_utils
import numpy as np
import pinocchio as pin
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

# CONFIG_NAME = 'config'
CONFIG_NAME = 'config36d'

CONFIG_PATH = CONFIG_NAME+".yml"
config      = path_utils.load_yaml_file(CONFIG_PATH)



data_path = '/home/skleff/Desktop/delta_f_real_exp/3d/integral/step/'
    
print("Load data 1...")
r1 = DataReader(data_path+'config36d_REAL_2023-09-11T15:26:12.838067_baseline.mds')  
print("Load data 2...")
r2 = DataReader(data_path+'config36d_REAL_2023-09-11T15:44:38.246985_FI_best.mds') 
print("Load data 3...")
r3 = DataReader(data_path+'config36d_REAL_2023-09-11T17:10:36.216511_DF_best.mds')

label1 = 'Default'
label2 = 'Integral'
label3 = 'Estimation'


FILTER = 1
from core_mpc import analysis_utils 

N = min(r1.data['tau'].shape[0], r2.data['tau'].shape[0])
N = min(N, r3.data['tau'].shape[0])

N_START = 0 # int(config['T_CIRCLE'] * config['simu_freq']) 
print("N_start = ", N_START)



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


target_force_3d = np.zeros((N, 3))
Tc = int(config['T_CONTACT'] * config['simu_freq'])
target_force_3d[Tc:, :3] = np.asarray(config['frameForceRef'])[:3]
N_ramp = int((config['T_RAMP'] - config['T_CONTACT']) * config['simu_freq'])           
FZ_MIN = 5. 
FZ_MAX = 100.
FX_MAX = config['frameForceRef'][0]
target_force_3d[Tc:Tc+N_ramp, 2] = [FZ_MIN + (FZ_MAX - FZ_MIN)*i/N_ramp for i in range(N_ramp)]
target_force_3d[Tc+N_ramp:, 2] = FZ_MAX
freq = 0.02
# target[Tc+N_ramp:, 2] = [FZ_MAX + 40.*np.round(freq * (2*np.pi) * i / config['simu_freq'] - int(freq * (2*np.pi) * i / config['simu_freq'])) for i in range(N-N_ramp-Tc)]
target_force_3d[Tc+N_ramp:, 0] = [FX_MAX + 20.*(np.round(freq * (2*np.pi) * i / config['simu_freq'] - int(freq * (2*np.pi) * i / config['simu_freq']))-0.5) for i in range(N-N_ramp-Tc)]



time_lin = np.linspace(0, (N-N_START) / config['simu_freq'], (N-N_START))


# Generate plot of number of iterations for each problem
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
 
    
    
fig0, ax = plt.subplots(3, 1, figsize=(10.8,10.8))

for i in range(3):
    ax[i].plot(time_lin, target_force_3d[:,i], color='k', linewidth=4, linestyle='-', label='Reference', alpha=1.) 
    ax[i].plot(time_lin, force_3d_1[:,i], color='b', linewidth=4, label=label1, alpha=0.5)
    # ax[i].plot(time_lin, force_3d_2[:,i], color='g', linewidth=4, label=label2, alpha=0.5)
    # ax[i].plot(time_lin, force_3d_3[:,i], color='r', linewidth=4, label=label3, alpha=0.5)
    
    
    ax[i].grid(True) 
ax[0].legend()    
plt.show()

ax0.set_ylabel('Z (m)', fontsize=26)
ax0.set_xlabel('Y (m)', fontsize=26)
ax0.tick_params(axis = 'y', labelsize=22)
ax0.tick_params(axis = 'x', labelsize=22)




 
# Plot end-effector trajectory (y,z) plane 
def plot_joint_traj(jmea, label):
    fig0, ax0 = plt.subplots(1, 1, figsize=(19.2,10.8))
    # Measured
    ax0.plot(xdata, jmea, color='b', linewidth=4, label=label, alpha=0.5) 
    # Axis label & ticks
    ax0.set_ylabel('Joint position $q_1$ (rad)', fontsize=26)
    ax0.set_xlabel('Time (s)', fontsize=26)
    ax0.tick_params(axis = 'y', labelsize=22)
    ax0.tick_params(axis = 'x', labelsize=22)
    ax0.grid(True) 
    return fig0, ax0


p_mea = get_p_(r1.data['joint_positions'][N_start:N], pinrobot.model, pinrobot.model.getFrameId('contact'))
fig0, ax0 = plot_endeff_yz(p_mea, target_position) 
ax0.set_xlim(-0.33, +0.33)
ax0.set_ylim(0.15, 0.8)
ax0.plot(p_mea[0,1], p_mea[0,2], 'ro', markersize=16)
ax0.text(0., 0.1, '$x_0$', fontdict={'size':26})
# handles, labels = ax0.get_legend_handles_labels()
# fig0.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.12, 0.885), prop={'size': 26}) 
# fig0.savefig('/home/skleff/data_paper_fadmm/no_cstr_circle_plot.pdf', bbox_inches="tight")
# Joint pos
jmea = r1.data['joint_positions'][N_start:N, 0]
fig1, ax1 = plot_joint_traj(jmea) 
ax1.set_ylim(-2., 2.)
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.12, 0.885), prop={'size': 26}) 
# fig.savefig('/home/skleff/data_paper_fadmm/no_cstr_q1_plot.pdf', bbox_inches="tight")



plt.show()
plt.close('all')