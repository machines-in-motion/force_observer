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
r1 = DataReader(data_path+'config36d_REAL_2023-09-13T16:39:19.602786_baseline.mds')  
print("Load data 2...")
r2 = DataReader(data_path+'config36d_REAL_2023-09-13T16:40:12.238629_DF_int.mds') 
print("Load data 3...")
r3 = DataReader(data_path+'config36d_REAL_2023-09-13T16:41:03.995670_DF_ext.mds') 
print("Load data 4...")
r4 = DataReader(data_path+'config36d_REAL_2023-09-13T16:41:57.325986_FI.mds')

label1 = 'Default'
label2 = 'Estimation Int'
label3 = 'Estimation Ext'
label4 = 'Integral'


FILTER = 1
from core_mpc import analysis_utils 

N = min(r1.data['tau'].shape[0], r2.data['tau'].shape[0])
N = min(N, r3.data['tau'].shape[0])

N_START = 9800 
N = 10800
print("N_start = ", N_START)




force_3d_1 = np.zeros((N-N_START, 3))
force_3d_2 = np.zeros((N-N_START, 3))
force_3d_3 = np.zeros((N-N_START, 3))
force_3d_4 = np.zeros((N-N_START, 3))



if(FILTER > 0):
    force_3d_1 = analysis_utils.moving_average_filter(r1.data['contact_force_3d_measured'][N_START:N].copy(), FILTER)
    force_3d_2 = analysis_utils.moving_average_filter(r2.data['contact_force_3d_measured'][N_START:N].copy(), FILTER) 
    force_3d_3 = analysis_utils.moving_average_filter(r3.data['contact_force_3d_measured'][N_START:N].copy(), FILTER) 
    force_3d_4 = analysis_utils.moving_average_filter(r4.data['contact_force_3d_measured'][N_START:N].copy(), FILTER) 
else:
    force_3d_1 = r1.data['contact_force_3d_measured'][N_START:N]
    force_3d_2 = r2.data['contact_force_3d_measured'][N_START:N] 
    force_3d_3 = r3.data['contact_force_3d_measured'][N_START:N] 
    force_3d_4 = r4.data['contact_force_3d_measured'][N_START:N] 


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
    ax[i].plot(time_lin, target_force_3d[N_START:N,i], color='k', linewidth=4, linestyle='--', label='Reference', alpha=0.5) 
    ax[i].plot(time_lin, force_3d_1[:,i], color='b', linewidth=4, label=label1, alpha=1.)
    ax[i].plot(time_lin, force_3d_2[:,i], color='g', linewidth=4, label=label2, alpha=1.)
    ax[i].plot(time_lin, force_3d_3[:,i], color='r', linewidth=4, label=label3, alpha=1.)
    ax[i].plot(time_lin, force_3d_4[:,i], color='y', linewidth=4, label=label4, alpha=1.)
    
    
    ax[i].grid(True) 
ax[0].legend() 
ax[0].set_ylabel('F (N)', fontsize=26)
ax[0].set_xlabel('Time (s)', fontsize=26)
ax[0].tick_params(axis = 'y', labelsize=22)
ax[0].tick_params(axis = 'x', labelsize=22)


fig1 = plt.figure(figsize=(10.8,10.8))

plt.plot(time_lin, target_force_3d[N_START:N, 0], color='k', linewidth=4, linestyle='--', label='Reference', alpha=0.5) 
plt.plot(time_lin, force_3d_1[:,0], color='b', linewidth=4, label=label1, alpha=1.)
plt.plot(time_lin, force_3d_2[:,0], color='g', linewidth=4, label=label2, alpha=1.)
plt.plot(time_lin, force_3d_3[:,0], color='r', linewidth=4, label=label3, alpha=1.)
plt.plot(time_lin, force_3d_4[:,0], color='y', linewidth=4, label=label4, alpha=1.)

plt.grid(True) 
plt.legend() 
plt.xlim(time_lin[0], time_lin[-1])
plt.ylabel('F (N)', fontsize=26)
plt.xlabel('Time (s)', fontsize=26)
plt.tick_params(axis = 'y', labelsize=22)
plt.tick_params(axis = 'x', labelsize=22)


def print_error(r, label):
    error = np.abs(r.data['contact_force_3d_measured'][N_START:N] - target_force_3d[N_START:N])
    print(label, " Fx mean abs error      = ", np.mean(error[:, 0]))
    print(label, " Fy mean abs error      = ", np.mean(error[:, 1]))
    print(label, " Fz mean abs error      = ", np.mean(error[:, 2]))
    print(label, " F mean abs error      = ",  np.mean(error))
    print('\n')

print_error(r1, label1)
print_error(r2, label2)
print_error(r3, label3)
print_error(r4, label4)

# fig.savefig('/home/skleff/data_paper_fadmm/no_cstr_q1_plot.pdf', bbox_inches="tight")



plt.show()
plt.close('all')