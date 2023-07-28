import sys
sys.path.append("/home/ajordana/ws/workspace/src/")
from force_feedback_dgh.demos.utils.plot_utils import SimpleDataPlotter
from mim_data_utils import DataReader
from core_mpc.pin_utils import *
from core_mpc import path_utils
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt 
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


# Create data Plotter
s = SimpleDataPlotter()

data_path = '/home/ajordana/Desktop/delta_f_real_exp/sanding/'
label1 = 'no_delta_f'
label2 = 'fric_only'
label3 = 'delta_f_Q_fric'

SAVE = False

print("Load data 1...")
r1 = DataReader(data_path+'config_REAL_2023-07-14T11:36:25.882687_no_delta_f.mds')  
print("Load data 2...")
r2 = DataReader('/home/ajordana/Desktop/delta_f_real_exp/filter/'+'config_REAL_2023-07-20T17:40:00.571261delta_f_Q=R=4e-3_fric.mds') 
print("Load data 3...")
r3 = DataReader('/home/ajordana/Desktop/delta_f_real_exp/filter/'+'config_REAL_2023-07-20T18:36:11.700721delta_f_fric_best.mds')


# Load config file
CONFIG_NAME = 'config.yml'
config      = path_utils.load_yaml_file(CONFIG_NAME)

FILTER = 100
from core_mpc import analysis_utils 


N = min(r1.data['tau'].shape[0], r2.data['tau'].shape[0])
N = min(N, r3.data['tau'].shape[0])

target_force_3d = np.zeros((N, 3))
target_force_3d[:,-1] = config['frameForceRef'][2]

N_START = int(config['T_CIRCLE'] * config['simu_freq'])
print("N_start = ", N_START)

if(FILTER > 0):
    r1.data['contact_force_3d_measured'][:N] = analysis_utils.moving_average_filter(r1.data['contact_force_3d_measured'][:N].copy(), FILTER)
    r2.data['contact_force_3d_measured'][:N] = analysis_utils.moving_average_filter(r2.data['contact_force_3d_measured'][:N].copy(), FILTER) 
    r3.data['contact_force_3d_measured'][:N] = analysis_utils.moving_average_filter(r3.data['contact_force_3d_measured'][:N].copy(), FILTER) 

fig_f1, _ = s.plot_soft_contact_force([
                           r1.data['contact_force_3d_measured'][N_START:N], 
                           r2.data['contact_force_3d_measured'][N_START:N], 
                           r3.data['contact_force_3d_measured'][N_START:N], 
                           target_force_3d[N_START:N]
                           ],
                           [
                            label1, 
                            label2,
                            label3,
                            'ref'
                          ], 
                          [
                            'b', 
                            'g', 
                            'r', 
                            'k'
                            ],
                          linestyle=[
                            'solid', 
                            'solid', 
                            'solid', 
                            'dotted'
                            ])


fig_f2, _ = s.plot_soft_contact_force([
                           np.abs(r1.data['contact_force_3d_measured'][N_START:N] - target_force_3d[N_START:N]), 
                           np.abs(r2.data['contact_force_3d_measured'][N_START:N] - target_force_3d[N_START:N]), 
                           np.abs(r3.data['contact_force_3d_measured'][N_START:N] - target_force_3d[N_START:N]), 
                           target_force_3d[N_START:N] - target_force_3d[N_START:N]
                           ],
                           [
                            label1,
                            label2, 
                            label3,
                            'ref'
                          ], 
                          [
                            'b', 
                            'g', 
                            'r', 
                            'k'
                            ],
                          linestyle=[
                            'solid', 
                            'solid', 
                            'solid', 
                            'dotted'
                            ])



p_mea1 = get_p_(r1.data['joint_positions'][N_START:N,controlled_joint_ids][:,controlled_joint_ids], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_mea2 = get_p_(r2.data['joint_positions'][N_START:N,controlled_joint_ids][:,controlled_joint_ids], pinrobot.model, pinrobot.model.getFrameId('contact'))
p_mea3 = get_p_(r3.data['joint_positions'][N_START:N,controlled_joint_ids][:,controlled_joint_ids], pinrobot.model, pinrobot.model.getFrameId('contact'))
target_position = np.zeros((N-N_START, 3))
target_position[:,0] = r1.data['target_position_x'][N_START:N,0]
target_position[:,1] = r1.data['target_position_y'][N_START:N,0]
target_position[:,2] = r1.data['target_position_z'][N_START:N,0]
target_position2 = np.zeros((N-N_START, 3))
target_position2[:,0] = r2.data['target_position_x'][N_START:N,0]
target_position2[:,1] = r2.data['target_position_y'][N_START:N,0]
target_position2[:,2] = r2.data['target_position_z'][N_START:N,0]
target_position3 = np.zeros((N-N_START, 3))
target_position3[:,0] = r3.data['target_position_x'][N_START:N,0]
target_position3[:,1] = r3.data['target_position_y'][N_START:N,0]
target_position3[:,2] = r3.data['target_position_z'][N_START:N,0]
fig_p1, _ = s.plot_ee_pos( [
                           p_mea1,
                           p_mea2 ,
                           p_mea3,
                           target_position 
                           ],
                           [
                            label1, 
                            label2, 
                            label3, 
                            'ref'
                          ], 
                          [
                            'b', 
                            'g', 
                            'r', 
                            'k'
                            ],
                          linestyle=[
                            'solid', 
                            'solid', 
                            'solid', 
                            'dotted'
                            ])

fig_p2, _ = s.plot_ee_pos( [
                           np.abs(p_mea1 - target_position),
                           np.abs(p_mea2 - target_position2),
                           np.abs(p_mea3 - target_position3),
                           target_position - target_position
                           ],
                           [
                            label1, 
                            label2,
                            label3, 
                            'ref'
                          ], 
                          [
                            'b', 
                            'g', 
                            'r', 
                            'k'
                            ],
                          linestyle=[
                            'solid', 
                            'solid', 
                            'solid', 
                            'dotted'
                            ])



fig, ax = plt.subplots(4, 1, sharex='col') 
ax[0].plot(r1.data['count']-1, label='count_'+label1)
ax[0].plot(r2.data['count']-1, label='count_'+label2)

ax[1].plot(r1.data['t_child'], label='child'+label1)
ax[1].plot(r2.data['t_child'], label='child'+label2)

ax[1].plot(r1.data['t_child_1'], label='child_1 (not solve)'+label1)
ax[1].plot(r2.data['t_child_1'], label='child_1 (not solve)'+label2)

ax[2].plot(r1.data['ddp_iter'], label='iter'+label1)
ax[2].plot(r2.data['ddp_iter'], label='iter'+label2)

ax[3].plot(r1.data['t_run'], label='t_run'+label1)
ax[3].plot(r1.data['t_run'], label='t_run'+label2)
plt.legend()


print("------------------------------------")
print("------------------------------------")
print(label1+" Pxy error norm = ", np.linalg.norm(p_mea1[N_START:N,:2] - target_position[N_START:N,:2]))
print(label2+" Pxy error norm = ", np.linalg.norm(p_mea2[N_START:N,:2] - target_position2[N_START:N,:2]))
print(label3+" Pxy error norm = ", np.linalg.norm(p_mea3[N_START:N,:2] - target_position3[N_START:N,:2]))
print(label1+" Fz error norm = ", np.linalg.norm(r1.data['contact_force_3d_measured'][N_START:N, 2] - target_force_3d[N_START:,2]))
print(label2+" Fz error norm = ", np.linalg.norm(r2.data['contact_force_3d_measured'][N_START:N, 2] - target_force_3d[N_START:,2]))
print(label3+" Fz error norm = ", np.linalg.norm(r3.data['contact_force_3d_measured'][N_START:N, 2] - target_force_3d[N_START:,2]))
print("------------------------------------")
print("------------------------------------")


if(SAVE):
  fig_f1.savefig(data_path+label1+'_vs_'+label2+'_vs_'+label3+'_compare_force.png')
  fig_f2.savefig(data_path+label1+'_vs_'+label2+'_vs_'+label3+'_compare_force_err.png')
  # fig_p1.savefig(data_path+label1+'_vs_'+label2+'_vs_'+label3'_compare_pos.png')
  fig_p2.savefig(data_path+label1+'_vs_'+label2+'_vs_'+label3+'_compare_pos_err.png')


plt.show()