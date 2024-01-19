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
CONFIG_NAME = 'polishing'
CONFIG_PATH = "config/"+CONFIG_NAME+".yml"
config      = path_utils.load_yaml_file(CONFIG_PATH)



# Load data 
SIM = False

# Create data Plottger
s = SimpleDataPlotter()



if(SIM):
    data_path = "/tmp/"
    data_name = 'polishing_SIM_2024-01-19T12:23:12.577464test'
    
else:
    data_path = '/tmp/'
    data_name = 'polishing_REAL_2024-01-19T18:31:38.701941test' 

r       = DataReader(data_path+data_name+'.mds')
print("\n reading "+data_path+data_name+'.mds\n')
N       = r.data['absolute_time'].shape[0]
print("Total number of control cycles = ", N)
time_lin = np.linspace(0, N/ config['plan_freq'], (N))

fig, ax = plt.subplots(4, 1, sharex='col') 
ax[0].plot(r.data['KKT'], label='KKT residual')
ax[0].plot(N*[config['solver_termination_tolerance']], label= 'KKT residual tolerance', color = 'r')
# ax[1].plot(r.data['ddp_iter'], label='# solver iterations')
ax[2].plot(r.data['t_child']*1000, label='OCP solve time')
ax[2].plot(N*[1000./config['ctrl_freq']], label= 'dt_MPC', color='r')
ax[3].plot((r.data['timing_control'])* 1000, label='Control cycle time')
ax[3].plot(N*[1000./config['ctrl_freq']], label= 'dt_MPC', color='r')
for i in range(4):
    ax[i].grid()
    ax[i].legend()

PHASE_TIME = 6283
N_TOTAL = 3 * PHASE_TIME
print("N_TOTAL", N_TOTAL * 1e-3)

FILTER = 10
N_START = int(config['T_CIRCLE'] * config['ctrl_freq']) 
print("N_start = ", N_START)

if(FILTER > 0):
    r.data['contact_force_3d_measured'][:N] = analysis_utils.moving_average_filter(r.data['contact_force_3d_measured'][:N].copy(), FILTER)

s.plot_joint_pos( [r.data['x_des'][:,:nq],
                   r.data['joint_positions'][:,controlled_joint_ids]],
                #    target_joint],
                   ['pred', 
                    'mea',
                    'ref'],
                   ['b', 
                    'r',
                    'k'],
                   linestyle=['solid', 'solid', 'dotted'],
                   ylims=[model.lowerPositionLimit, model.upperPositionLimit] )

if(config['USE_DELTA_F']):
    plt.figure()
    plt.plot(np.array(r.data['delta_f']))
    plt.title("delta_f")

if(config['USE_DELTA_TAU']):
    s.plot_joint_tau( [r.data['delta_tau']], 
                      ['delta_tau'], 
                      ['r'])

# For SIM robot only
if(SIM):
    s.plot_joint_tau( [r.data['tau'], 
                       r.data['tau_ff'], 
                       r.data['tau_riccati'], 
                       r.data['tau_gravity']],
                      ['total', 
                       'ff', 
                       'riccati', 
                       'gravity'], 
                      ['r', 
                       'g', 
                       'b', 
                       [0.2, 0.2, 0.2, 0.5]],
                      ylims=[-model.effortLimit, +model.effortLimit] )
# For REAL robot only !! DEFINITIVE FORMULA !!
else:
    # Our self.tau was subtracted gravity, so we add it again
    # joint_torques_measured DOES include the gravity torque from KUKA
    # There is a sign mismatch in the axis so we use a minus sign
    s.plot_joint_tau( [-r.data['joint_cmd_torques'][:,controlled_joint_ids], 
                       r.data['joint_torques_measured'][:,controlled_joint_ids], 
                       r.data['tau'][:,controlled_joint_ids] + r.data['tau_gravity'][:,controlled_joint_ids],
                       r.data['tau_ff'][:,controlled_joint_ids] + r.data['tau_gravity'][:,controlled_joint_ids],
                     r.data['tau_gravity'][:,controlled_joint_ids]],
                    #    r.data['joint_ext_torques'][:,controlled_joint_ids]], 
                  ['-cmd (FRI)', 
                   'Measured', 
                   'Desired (sent to robot) [+g(q)]', 
                   'tau_ff (OCP solution) [+g(q)]', 
                   'g(q)',
                   'EXT'], 
                  ['k', 'r', 'b', 'g', 'y'],
                  ylims=[-model.effortLimit, +model.effortLimit],
                  linestyle=['dotted', 'solid', 'solid', 'solid', 'solid'])


p_mea = pin_utils.get_p_(r.data['joint_positions'][:,controlled_joint_ids], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_des = pin_utils.get_p_(r.data['x_des'][:,:nq], pinrobot.model, pinrobot.model.getFrameId('contact'))
target_position = np.zeros((N, 3))
target_position[:,0] = r.data['target_position_x'][:,0]
target_position[:,1] = r.data['target_position_y'][:,0]
target_position[:,2] = r.data['target_position_z'][:,0]
fig_p, _ = s.plot_ee_pos( [p_mea, 
                target_position],  
               ['mea', 'ref (position cost)'], 
               ['r',  'k'], 
               linestyle=['solid', 'dotted'])

plt.figure(figsize=(20, 10))
plt.plot(r.data['absolute_time'], p_mea[:,0], 'r', label='x pos Measured')
plt.plot(r.data['absolute_time'], target_position[:, 0], 'k', label='x pos ref')
plt.grid()
plt.legend()



target_force_3d = np.zeros((N, 3))
target_force_3d[:,0] = r.data['target_force'][:,0]*0.
target_force_3d[:,1] = r.data['target_force'][:,0]*0.
target_force_3d[:,2] = r.data['target_force'][:,0]

force_delta_f = np.zeros((N, 3))
force_delta_f[:,2] = np.array(r.data['contact_force_3d_measured'][:,2]) + np.array(r.data['delta_f'])[:,0]


target = np.zeros((N, 3))
Tc = int(config['T_CONTACT'] * config['ctrl_freq'])
target[Tc:, :3] = np.asarray(config['frameForceRef'])[:3]
N_ramp = int((config['T_RAMP'] - config['T_CONTACT']) * config['ctrl_freq'])            # Ref in Fz
FZ_MIN = 5. #self.config['frameForceRef'][2] #0.
FZ_MAX = config['frameForceRef'][2]
FX_MAX = config['frameForceRef'][0]
target[Tc:Tc+N_ramp, 2] = [FZ_MIN + (FZ_MAX - FZ_MIN)*i/N_ramp for i in range(N_ramp)]
target[Tc+N_ramp:, 2] = FZ_MAX




# Plot forces
fig_f, _ = s.plot_soft_contact_force([
                           r.data['contact_force_3d_measured'][:,:3], 
                           r.data['force_est'], 
                           target_force_3d,
                           target],
                          ['Measured', 
                           'Filtered',
                           'Target (modified)', 
                           'Target',
                           'Predicted'], 
                          ['r', 'g', 'b', 'k'],
                          linestyle=['solid', 'solid', 'solid', 'dotted', 'dotted'])#,
                        #   ylims=[[-50,-50, 0], [50, 50, 1070]])


plt.figure(figsize=(20, 10))
plt.plot(r.data['absolute_time'], r.data['contact_force_3d_measured'][:,2], 'r', label='Measured')
plt.plot(r.data['absolute_time'], r.data['force_est'][:,2], 'g', label='Filtered')
plt.plot(r.data['absolute_time'], target_force_3d[:, 2], 'b', label='Target (modified)')
plt.plot(r.data['absolute_time'], target[:, 2], 'k', label='Target')
plt.grid()
plt.legend()



# Compute average tracking error for each circle
CIRCLE_PERIOD_IN_CYCLES = int(2*np.pi/3.*1000)
N_START = N_START
N_CIRCLE = int((N-N_START)/CIRCLE_PERIOD_IN_CYCLES)
N = N_START + CIRCLE_PERIOD_IN_CYCLES * N_CIRCLE

error_traj = np.abs(r.data['contact_force_3d_measured'][N_START:, 2] - target[N_START:, 2])
mean_errors = [np.mean(error_traj[t*CIRCLE_PERIOD_IN_CYCLES:(t+1)*CIRCLE_PERIOD_IN_CYCLES]) for t in range(N_CIRCLE)]
print(mean_errors)

print(" Fz mean abs error [1:] = ", np.mean(mean_errors[1:]), r'$\pm$', np.std(mean_errors[1:]))
print(" Fz mean abs error      = ", np.mean(mean_errors), r'$\pm$', np.std(mean_errors))
# print(" F3d mean abs error = ", np.mean(np.linalg.norm(np.abs(r.data['contact_force_3d_measured'][N_START:N, :] - target[N_START:, :]))))


error_traj_pos = np.abs(p_mea[N_START:, :2] - target_position[N_START:, :2])
mean_error_pos = [np.mean(error_traj_pos[t*CIRCLE_PERIOD_IN_CYCLES:(t+1)*CIRCLE_PERIOD_IN_CYCLES]) for t in range(1, N_CIRCLE)]
# print(mean_errors)
print( ": POS mean abs error      = ",  np.mean(mean_error_pos), "  +-  ", np.std(mean_error_pos))
print('\n')    

# if(SAVE):
#     fig_f.savefig(data_path+data_name+'_force.png')
#     fig_p.savefig(data_path+data_name+'_pos.png')


plt.show()