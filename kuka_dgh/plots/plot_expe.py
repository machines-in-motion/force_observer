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

# target_joint = np.zeros((N,nq))
# target_joint[:, 2] = r.data['target_joint'][:,0]

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
if CONFIG_NAME == 'normal_force':
    target_force_3d[:,0] = r.data['target_force_fx'][:,0]
    target_force_3d[:,1] = r.data['target_force_fy'][:,0]
    target_force_3d[:,2] = r.data['target_force_fz'][:,0]
else:
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
# if CONFIG_NAME == 'config36d':
#     freq = 0.02
#     # target[Tc+N_ramp:, 2] = [FZ_MAX + 40.*np.round(freq * (2*np.pi) * i / config['ctrl_freq'] - int(freq * (2*np.pi) * i / config['ctrl_freq'])) for i in range(N-N_ramp-Tc)]
#     target[Tc+N_ramp:, 2] = [FZ_MAX + 50.*(np.round(freq * (2*np.pi) * i / config['ctrl_freq'] - int(freq * (2*np.pi) * i / config['ctrl_freq']))-0.5) for i in range(N-N_ramp-Tc)]





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



if CONFIG_NAME == 'polishing':
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
    
    
else:
    print(" F_xy mean abs error      = ", np.mean(np.abs(r.data['contact_force_3d_measured'][N_START:N, :2] - target[N_START:N, :2])))
    print(" Fz mean abs error      = ", np.mean(np.abs(r.data['contact_force_3d_measured'][N_START:N, 2] - target[N_START:N, 2])))
    print(" F mean abs error      = ", np.mean(np.abs(r.data['contact_force_3d_measured'][N_START:N] - target[N_START:N])))



    
    
# Compute energy
if CONFIG_NAME == 'normal_force':
    if SIM:
        print("Total energy = ",  np.mean([np.linalg.norm(u) for u in r.data['tau'][N_START:, controlled_joint_ids]]))
    else:
        torque_list = [np.linalg.norm(u1+u2) for u1, u2 in zip(r.data['tau'][N_START:, controlled_joint_ids], r.data['tau_gravity'][:,controlled_joint_ids])]
        print("Avg Torque = ",  np.mean([np.linalg.norm(u) for u in torque_list]) )
        
        torque_square_norm = [np.linalg.norm(u)**2 for u in torque_list]
        print("Avg Square Torque = ",  np.mean(torque_square_norm) )

        CIRCLE_PERIOD_IN_CYCLES = int(1000 / 0.5)
        print(CIRCLE_PERIOD_IN_CYCLES)
        N_CIRCLE = int((N-N_START)/CIRCLE_PERIOD_IN_CYCLES)
        N = N_START + CIRCLE_PERIOD_IN_CYCLES * N_CIRCLE
        print(N_CIRCLE)
        
        def mean_std(np_array):
            np_array = [np.mean(np_array[t*CIRCLE_PERIOD_IN_CYCLES:(t+1)*CIRCLE_PERIOD_IN_CYCLES]) for t in range(1, N_CIRCLE)]
            return np.mean(np_array), np.std(np_array)
                    
        mean_torque_errors = [np.mean(torque_square_norm[t*CIRCLE_PERIOD_IN_CYCLES:(t+1)*CIRCLE_PERIOD_IN_CYCLES]) for t in range(N_CIRCLE)]
        # print("Avg Square Torque [1:] = ", np.mean(mean_torque_errors[1:]), r'$\pm$', np.std(mean_torque_errors[1:]))
        m , std = mean_std(torque_square_norm)
        print("Avg Square Torque [1:] = ", m, r'$\pm$', std)
        # print("Avg Square Torque      = ", np.mean(mean_torque_errors), r'$\pm$', np.std(mean_torque_errors))




    cost_list = []
    force_cost_list = []
    state_cost_list = []
    tau_cost_list = []
    rotation_cost_list = []
    for t in range(N- N_START):
        index = N_START  + t

        f = r.data['contact_force_3d_measured'][index][:3]   
        q = r.data['joint_positions'][index,controlled_joint_ids]
        v = r.data['joint_velocities'][index,controlled_joint_ids]
        x = np.concatenate([q, v])
        if SIM:
            tau = r.data['tau'][index, controlled_joint_ids] 
        else:
            tau = r.data['tau'][index, controlled_joint_ids] + r.data['tau_gravity'][index,controlled_joint_ids]

        rotation = pin.utils.rpyToMatrix(get_rpy_(r.data['joint_positions'][index,controlled_joint_ids], pinrobot.model, pinrobot.model.getFrameId('contact')))

        f_ref = config['frameForceRef'][:3]
        xref = np.array([0., 1.0471975511965976, r.data['target_joint'][index, 0], -1.1344640137963142, 0.2,  0.7853981633974483, 0.,0.,0.,0.,0.,0.])
        tau_ref = np.zeros(tau.shape)
        rotation_ref = pin.utils.rpyToMatrix(np.pi, 0, np.pi)



        force_cost = 0.5 * config['frameForceWeight'] * (f - f_ref).T @ np.diag(config['frameForceWeights'][:3])@ (f - f_ref)
        # force_cost = 0.5 * config['frameForceWeight'] * (f - f_ref).T @ np.diag(np.array([0, 0., 0.01]))@ (f - f_ref)
        force_cost_list.append(force_cost)


        state_cost = 0.5 * config['stateRegWeight'] * (x - xref).T @ np.diag(config['stateRegWeights'])@ (x - xref)
        state_cost_list.append(state_cost)

        tau_cost =  0.5 * 0.0008 * (tau - tau_ref).T @ (tau - tau_ref)
        tau_cost_list.append(tau_cost)


        rot_residual = pin.log3(rotation_ref.T @ rotation) 
        rotation_cost = 0.5 * config['frameRotationWeight'] * rot_residual.T @ np.diag(config['frameRotationWeights']) @ rot_residual
        rotation_cost_list.append(rotation_cost)


        cost_list.append(force_cost + state_cost + tau_cost + rotation_cost)



    cost_list = np.array(cost_list)
    force_cost_list = np.array(force_cost_list)
    state_cost_list = np.array(state_cost_list)
    tau_cost_list = np.array(tau_cost_list)
    rotation_cost_list = np.array(rotation_cost_list)



    if not SIM:
        mean , std = mean_std(cost_list)
        print("total cost ", mean, ' +- ', std)


        mean , std = mean_std(force_cost_list)
        print("force cost ", mean, ' +- ', std)


        mean , std = mean_std(state_cost_list)
        print("state cost ", mean, ' +- ', std)

        mean , std = mean_std(tau_cost_list)
        print("tau cost ", mean, ' +- ', std)


        mean , std = mean_std(rotation_cost_list)
        print("rotation cost ", mean, ' +- ', std)


    plt.figure()
    plt.plot(cost_list, label="total cost")
    plt.plot(force_cost_list, label="force cost")
    plt.plot(state_cost_list, label="state cost")
    plt.plot(tau_cost_list, label="tau cost")
    plt.plot(rotation_cost_list, label="rotation cost")
    plt.legend()
    plt.show()


# if(SAVE):
#     fig_f.savefig(data_path+data_name+'_force.png')
#     fig_p.savefig(data_path+data_name+'_pos.png')


plt.show()