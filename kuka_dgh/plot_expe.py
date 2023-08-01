import sys
sys.path.append("/home/ajordana/ws/workspace/src/")
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
CONFIG_PATH = CONFIG_NAME+".yml"
config      = path_utils.load_yaml_file(CONFIG_PATH)


# Load data 
SIM = False 
SAVE = False

# Create data Plottger
s = SimpleDataPlotter()

FILTER = 100


if(SIM):
    data_path = '/home/ajordana/Desktop/delta_f_real_exp/sanding_no_filter/'
    data_name = 'config_SIM_2023-08-01T13:58:50.797096.mds'
    
else:
    data_path = '/home/ajordana/Desktop/delta_f_real_exp/sanding_no_filter/'
    data_name = 'config_REAL_2023-08-01T16:26:44.658684_friction+df(no-acc-in-FK)_ter.mds'
    
# data_path = '/home/skleff/Desktop/soft_contact_real_exp/paper+video_datasets/slow/'
# data_name = 'reduced_soft_mpc_contact1d_REAL_2023-07-07T14:09:22.468998_slow_exp_2'

r = DataReader(data_path+data_name)
N = r.data['tau'].shape[0]

fig, ax = plt.subplots(4, 1, sharex='col') 
ax[0].plot(r.data['t_child'], label='t_solve')
ax[0].plot(N*[1./config['plan_freq']], label= 'mpc')
ax[1].plot(r.data['time_df'], label='DF_time')
ax[1].plot(N*[1./config['plan_freq']], label= 'mpc')
ax[2].plot(r.data['ddp_iter'], label='ddp iter')
ax[3].plot(r.data['t_run'], label='t_run')
ax[3].plot(N*[1./config['plan_freq']], label= 'mpc')
ax[0].legend(); ax[1].legend(); ax[2].legend(); ax[3].legend()

if(FILTER > 0):
    r.data['contact_force_3d_measured'][:N] = analysis_utils.moving_average_filter(r.data['contact_force_3d_measured'][:N].copy(), FILTER)


s.plot_joint_pos( [r.data['joint_positions'][:,controlled_joint_ids], r.data['x_des'][:,:nq]], # r.data['x'][:,:nq], r.data['x1'][:,:nq]], 
                   ['mea', 'pred'], #, 'pred0', 'pred1'], 
                   ['r', 'b'], #[0.2, 0.2, 0.2, 0.5], 'b', 'g'])
                #    markers=[None, None, '.', '.']) 
                   ylims=[model.lowerPositionLimit, model.upperPositionLimit] )
s.plot_joint_vel( [r.data['joint_velocities'][:,controlled_joint_ids], r.data['x_des'][:,nq:nq+nv]], # r.data['x'][:,nq:nq+nv], r.data['x1'][:,nq:nq+nv]],
                  ['mea', 'pred'], # 'pred0', 'pred1'], 
                  ['r', 'b'], #[0.2, 0.2, 0.2, 0.5], 'b', 'g']) 
                  ylims=[-model.velocityLimit, +model.velocityLimit] )


s.plot_joint_vel( [r.data['a'][:,controlled_joint_ids]],
                  ['mea', ], # 'pred0', 'pred1'], 
                  ['r'] )

plt.figure()
plt.plot(np.array(r.data['delta_f']))
plt.title("delta_f")

# For SIM robot only
if(SIM):
    s.plot_joint_tau( [r.data['tau'], r.data['tau_ff'], r.data['tau_riccati'], r.data['tau_gravity']], 
                      ['total', 'ff', 'riccati', 'gravity'], 
                      ['r', 'g', 'b', [0.2, 0.2, 0.2, 0.5]],
                      ylims=[-model.effortLimit, +model.effortLimit] )
# For REAL robot only !! DEFINITIVE FORMULA !!
else:
    # Our self.tau was subtracted gravity, so we add it again
    # joint_torques_measured DOES include the gravity torque from KUKA
    # There is a sign mismatch in the axis so we use a minus sign
    s.plot_joint_tau( [-r.data['joint_cmd_torques'][:,controlled_joint_ids], 
                       r.data['joint_torques_measured'][:,controlled_joint_ids], 
                       r.data['tau'][:,controlled_joint_ids] + r.data['tau_gravity'][:,controlled_joint_ids],
                       r.data['joint_ext_torques'][:,controlled_joint_ids]], 
                  ['-cmd (FRI)', 'Measured', 'Desired (+g(q))', 'EXT'], 
                  [[0.,0.,0.,0.], 'g', 'b', 'y'],
                  ylims=[-model.effortLimit, +model.effortLimit] )


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

# # # v_mea = get_v_(r.data['joint_velocities'], r.data['x_des'][:,nq:nq+nv], pinrobot.model, pinrobot.model.getFrameId('contact'))
# # # v_des = get_v_(r.data['joint_velocities'], r.data['x_des'][:,nq:nq+nv], pinrobot.model, pinrobot.model.getFrameId('contact'))
# # # target_velocity = np.zeros((N, 3))
# # # target_velocity[:,0] = r.data['target_velocity_x'][:,0]
# # # target_velocity[:,1] = r.data['target_velocity_y'][:,0]
# # # target_velocity[:,2] = r.data['target_velocity_z'][:,0]


target_force_3d = np.zeros((N, 3))
target_force_3d[:,0] = r.data['target_force'][:,0]*0
target_force_3d[:,1] = r.data['target_force'][:,0]*0
target_force_3d[:,2] = r.data['target_force'][:,0]


force_delta_f = np.zeros((N, 3))
force_delta_f[:,2] = np.array(r.data['contact_force_3d_measured'][:,2]) + np.array(r.data['delta_f'])[:,0]


# Plot forces
fig_f, _ = s.plot_soft_contact_force([
                           r.data['contact_force_3d_measured'], 
                           target_force_3d,
                        #    force_delta_f,
                           r.data['fpred']],
                          ['Measured', 'Reference',
                        #    'Measured+df', 
                           'Predicted'], 
                          ['r', 'b', 'k', 'g'],
                          linestyle=['solid', 'dotted', 'solid', 'solid'])
                        #   ylims=[[-50,-50, 0], [50, 50, 100]])


if(SAVE):
    fig_f.savefig(data_path+data_name+'_force.png')
    fig_p.savefig(data_path+data_name+'_pos.png')


plt.show()