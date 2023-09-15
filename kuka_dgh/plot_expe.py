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
# CONFIG_NAME = 'config'
CONFIG_NAME = 'config36d'

CONFIG_PATH = CONFIG_NAME+".yml"
config      = path_utils.load_yaml_file(CONFIG_PATH)


# Load data 
SIM = False 
SAVE = False

# Create data Plottger
s = SimpleDataPlotter()

FILTER = 1

N_START = int(config['T_CIRCLE'] * config['simu_freq']) 
print("N_start = ", N_START)

if(SIM):
    data_path = '/home/skleff/Desktop/delta_f_real_exp/3d/integral/energy_dtau/'
    data_name = 'config36d_SIM_2023-09-15T11:27:54.940770_test.mds'
    
else:
    data_path =  '/home/skleff/Desktop/delta_f_real_exp/3d/integral/energy_dtau/'
    data_name = 'config36d_REAL_2023-09-15T13:45:33.093715_dtau_ext_tune.mds'
    
# data_path = '/home/skleff/Desktop/soft_contact_real_exp/paper+video_datasets/slow/'
# data_name = 'reduced_soft_mpc_contact1d_REAL_2023-07-07T14:09:22.468998_slow_exp_2'

r = DataReader(data_path+data_name)
N = r.data['tau'].shape[0]

# fig, ax = plt.subplots(4, 1, sharex='col') 
# ax[0].plot(r.data['KKT'], label='KKT residual')
# # ax[0].plot(N*[1./config['plan_freq']], label= 'mpc')
# ax[1].plot(r.data['time_df'], label='DF_time')
# ax[1].plot(N*[1./config['plan_freq']], label= 'mpc')
# ax[2].plot(r.data['ddp_iter'], label='ddp iter')
# ax[3].plot(r.data['t_run'], label='t_run')
# ax[3].plot(N*[1./config['plan_freq']], label= 'mpc')
# ax[0].legend(); ax[1].legend(); ax[2].legend(); ax[3].legend()

if(FILTER > 0):
    r.data['contact_force_3d_measured'][:N] = analysis_utils.moving_average_filter(r.data['contact_force_3d_measured'][:N].copy(), FILTER)

# target_joint = np.zeros((N,nq))
# target_joint[:, 2] = r.data['target_joint'][:,0]

# s.plot_joint_pos( [r.data['x_des'][:,:nq],
#                    r.data['joint_positions'][:,controlled_joint_ids],
#                    target_joint],
#                    ['pred', 
#                     'mea',
#                     'ref'],
#                    ['b', 
#                     'r',
#                     'k'],
#                    linestyle=['solid', 'solid', 'dotted'],
#                    ylims=[model.lowerPositionLimit, model.upperPositionLimit] )
# s.plot_joint_vel( [r.data['joint_velocities'][:,controlled_joint_ids], r.data['x_des'][:,nq:nq+nv]], # r.data['x'][:,nq:nq+nv], r.data['x1'][:,nq:nq+nv]],
#                   ['mea', 'pred'], # 'pred0', 'pred1'], 
#                   ['r', 'b'], #[0.2, 0.2, 0.2, 0.5], 'b', 'g']) 
#                   ylims=[-model.velocityLimit, +model.velocityLimit] )


# s.plot_joint_vel( [r.data['a'][:,controlled_joint_ids],
#                    r.data['acc_est'][:,controlled_joint_ids]],
#                   ['a mea', 'a smooth' ], # 'pred0', 'pred1'], 
#                   ['r', 'c'] )

if(config['USE_DELTA_F']):
    plt.figure()
    plt.plot(np.array(r.data['delta_f']))
    plt.title("delta_f")

if(config['USE_DELTA_TAU']):
    s.plot_joint_tau( [r.data['delta_tau']], 
                      ['delta_tau'], 
                      ['r'])

# For SIM robot only
# if(SIM):
#     s.plot_joint_tau( [r.data['tau'], 
#                        r.data['tau_ff'], 
#                        r.data['tau_riccati'], 
#                        r.data['tau_gravity']],
#                       ['total', 
#                        'ff', 
#                        'riccati', 
#                        'gravity'], 
#                       ['r', 
#                        'g', 
#                        'b', 
#                        [0.2, 0.2, 0.2, 0.5]],
#                       ylims=[-model.effortLimit, +model.effortLimit] )
# # For REAL robot only !! DEFINITIVE FORMULA !!
# else:
#     # Our self.tau was subtracted gravity, so we add it again
#     # joint_torques_measured DOES include the gravity torque from KUKA
#     # There is a sign mismatch in the axis so we use a minus sign
#     s.plot_joint_tau( [-r.data['joint_cmd_torques'][:,controlled_joint_ids], 
#                        r.data['joint_torques_measured'][:,controlled_joint_ids], 
#                        r.data['tau'][:,controlled_joint_ids] + r.data['tau_gravity'][:,controlled_joint_ids],
#                        r.data['tau_ff'][:,controlled_joint_ids] + r.data['tau_gravity'][:,controlled_joint_ids],
#                      r.data['tau_gravity'][:,controlled_joint_ids]],
#                     #    r.data['joint_ext_torques'][:,controlled_joint_ids]], 
#                   ['-cmd (FRI)', 
#                    'Measured', 
#                    'Desired (sent to robot) [+g(q)]', 
#                    'tau_ff (OCP solution) [+g(q)]', 
#                    'g(q)',
#                    'EXT'], 
#                   ['k', 'r', 'b', 'g', 'y'],
#                   ylims=[-model.effortLimit, +model.effortLimit],
#                   linestyle=['dotted', 'solid', 'solid', 'solid', 'solid'])


p_mea = get_p_(r.data['joint_positions'][:,controlled_joint_ids], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_des = get_p_(r.data['x_des'][:,:nq], pinrobot.model, pinrobot.model.getFrameId('contact'))
target_position = np.zeros((N, 3))
target_position[:,0] = r.data['target_position_x'][:,0]
target_position[:,1] = r.data['target_position_y'][:,0]
target_position[:,2] = r.data['target_position_z'][:,0]
fig_p, _ = s.plot_ee_pos( [p_mea, 
                target_position],  
               ['mea', 'ref (position cost)'], 
               ['r',  'k'], 
               linestyle=['solid', 'dotted'])

# v_mea = get_v_(r.data['joint_velocities'], r.data['x_des'][:,nq:nq+nv], pinrobot.model, pinrobot.model.getFrameId('contact'))



target_force_3d = np.zeros((N, 3))
if CONFIG_NAME == 'config36d':
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
Tc = int(config['T_CONTACT'] * config['simu_freq'])
target[Tc:, :3] = np.asarray(config['frameForceRef'])[:3]
N_ramp = int((config['T_RAMP'] - config['T_CONTACT']) * config['simu_freq'])            # Ref in Fz
FZ_MIN = 5. #self.config['frameForceRef'][2] #0.
FZ_MAX = config['frameForceRef'][2]
FX_MAX = config['frameForceRef'][0]
target[Tc:Tc+N_ramp, 2] = [FZ_MIN + (FZ_MAX - FZ_MIN)*i/N_ramp for i in range(N_ramp)]
target[Tc+N_ramp:, 2] = FZ_MAX
# if CONFIG_NAME == 'config36d':
#     freq = 0.02
#     # target[Tc+N_ramp:, 2] = [FZ_MAX + 40.*np.round(freq * (2*np.pi) * i / config['simu_freq'] - int(freq * (2*np.pi) * i / config['simu_freq'])) for i in range(N-N_ramp-Tc)]
#     target[Tc+N_ramp:, 2] = [FZ_MAX + 50.*(np.round(freq * (2*np.pi) * i / config['simu_freq'] - int(freq * (2*np.pi) * i / config['simu_freq']))-0.5) for i in range(N-N_ramp-Tc)]





# Plot forces
r.data['compensation'] = np.zeros((N,3))
fig_f, _ = s.plot_soft_contact_force([
                           r.data['compensation'][:,:3], 
                           r.data['contact_force_3d_measured'][:,:3], 
                           r.data['force_est'], 
                           target_force_3d,
                           target],
                          ['comp', 
                           'Measured', 
                           'Filtered',
                           'Target (modified)', 
                           'Target',
                           'Predicted'], 
                          ['y', 'r', 'g', 'b', 'k'],
                          linestyle=['solid', 'solid', 'solid', 'dotted', 'dotted'])#,
                        #   ylims=[[-50,-50, 0], [50, 50, 1070]])
# plot force integral
# plt.figure()
# plt.plot(r.data['force_integral'])


if CONFIG_NAME == 'config':
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
if CONFIG_NAME == 'config36d':
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

        tau_cost =  0.5 * 0.002 * (tau - tau_ref).T @ (tau - tau_ref)
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


if(SAVE):
    fig_f.savefig(data_path+data_name+'_force.png')
    fig_p.savefig(data_path+data_name+'_pos.png')


plt.show()