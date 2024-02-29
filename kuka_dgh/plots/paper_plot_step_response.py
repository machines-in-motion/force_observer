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
CONFIG_NAME = 'normal_force'
CONFIG_PATH = "config/"+CONFIG_NAME+".yml"
config      = path_utils.load_yaml_file(CONFIG_PATH)



data_path = '/home/skleff/Desktop/ICRA24/dataset_force_offset_ICRA_2024/3d/integral/step_final/'
print("Load data 1...")
r1 = DataReader(data_path+'config36d_REAL_2023-09-13T17_55_41.465521_baseline.mds')  
print("Load data 2...")
r2 = DataReader(data_path+'config36d_REAL_2023-09-13T17_54_54.221924_df_int.mds') 
print("Load data 3...")
r3 = DataReader(data_path+'config36d_REAL_2023-09-13T17_53_02.723577_df_ext.mds') 
print("Load data 4...")
r4 = DataReader(data_path+'config36d_REAL_2023-09-13T17_54_03.901817_FI.mds')

label1 = 'Default'
label2 = 'Estimation (Model)'
label3 = 'Estimation (Feedforward)'
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


fig1 = plt.figure(figsize=(20,10))

plt.plot(time_lin, target_force_3d[N_START:N, 0], color='k', linewidth=4, linestyle='--', label='Reference', alpha=0.5) 
plt.plot(time_lin, force_3d_1[:,0], color='b', linewidth=4, label=label1, alpha=1.)
plt.plot(time_lin, force_3d_2[:,0], color='g', linewidth=4, label=label2, alpha=1.)
plt.plot(time_lin, force_3d_3[:,0], color='r', linewidth=4, label=label3, alpha=1.)
plt.plot(time_lin, force_3d_4[:,0], color='y', linewidth=4, label=label4, alpha=1.)

plt.grid(True) 
plt.legend(fontsize=26) 
plt.xlim(time_lin[0], time_lin[-1])
plt.ylabel('F (N)', fontsize=26)
plt.xlabel('Time (s)', fontsize=26)
plt.tick_params(axis = 'y', labelsize=22)
plt.tick_params(axis = 'x', labelsize=22)

# if(SAVE):
#     fig1.savefig('/home/skleff/Desktop/delta_f_real_exp/3d/integral/step/step_response.pdf', bbox_inches="tight")


# Generate Table III of the paper
def print_error(r, label):
    # Compute average tracking error for each circle
    error_x = np.abs(r.data['contact_force_3d_measured'][N_START:N, 0] - target_force_3d[N_START:N, 0])
    error_y = np.abs(r.data['contact_force_3d_measured'][N_START:N, 1] - target_force_3d[N_START:N, 1])
    error_z = np.abs(r.data['contact_force_3d_measured'][N_START:N, 2] - target_force_3d[N_START:N, 2])
    print(label, " Fx mean abs error      = ", np.mean(error_x))#, r'$\pm$', np.std(error_x))
    print(label, " Fy mean abs error      = ", np.mean(error_y))#, r'$\pm$', np.std(error_y))
    print(label, " Fz mean abs error      = ", np.mean(error_z))#, r'$\pm$', np.std(error_z))
    error = np.abs(r.data['contact_force_3d_measured'][N_START:N] - target_force_3d[N_START:N])
    error = np.mean(error, axis=1)
    # print("error = ", error)
    print(label, " F3d mean abs error      = ", np.mean(error)) #, r'$\pm$', np.std(error))
    print('\n')


print_error(r1, label1)
print_error(r2, label2)
print_error(r3, label3)
print_error(r4, label4)

plt.show()
plt.close('all')