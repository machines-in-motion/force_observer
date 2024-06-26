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

data_path = '/home/skleff/Desktop/delta_f_real_exp/3d/integral/'
label1 = 'FI_tune'
label2 = 'DF_tune'
label3 = 'DF'

SAVE = False

print("Load data 1...")
r1 = DataReader(data_path+'config36d_REAL_2023-09-11T18:26:46.367469_FI_tune.mds')  
print("Load data 2...")
r2 = DataReader(data_path+'config36d_REAL_2023-09-11T18:17:26.896803_DF_tune.mds') 
print("Load data 3...")
r3 = DataReader(data_path+'config36d_REAL_2023-09-11T18:08:20.416763_DF.mds')


# Load config file
# CONFIG_NAME = 'config.yml'
CONFIG_NAME = 'config36d.yml'
config      = path_utils.load_yaml_file(CONFIG_NAME)

FILTER = 1
from core_mpc import analysis_utils 


N = min(r1.data['tau'].shape[0], r2.data['tau'].shape[0])
# N = min(N, r3.data['tau'].shape[0])

plt.figure()
plt.plot(r1.data['ddp_iter'], label=label1)
plt.plot(r2.data['ddp_iter'], label=label2)
plt.legend()
# ax[0].plot(N*[1./config['plan_freq']], label= 'mpc')

# target_force_3d = np.zeros((N, 3))
# target_force_3d[:,-1] = config['frameForceRef'][2]


target_force_3d = np.zeros((N, 3))
Tc = int(config['T_CONTACT'] * config['simu_freq'])
target_force_3d[Tc:, :3] = np.asarray(config['frameForceRef'])[:3]
N_ramp = int((config['T_RAMP'] - config['T_CONTACT']) * config['simu_freq'])            # Ref in Fz
FZ_MIN = 5. #self.config['frameForceRef'][2] #0.
FZ_MAX = config['frameForceRef'][2]
FX_MAX = config['frameForceRef'][0]
target_force_3d[Tc:Tc+N_ramp, 2] = [FZ_MIN + (FZ_MAX - FZ_MIN)*i/N_ramp for i in range(N_ramp)]
target_force_3d[Tc+N_ramp:, 2] = FZ_MAX
# if CONFIG_NAME == 'config36d.yml':
#     freq = 0.02
#     # target[Tc+N_ramp:, 2] = [FZ_MAX + 40.*np.round(freq * (2*np.pi) * i / config['simu_freq'] - int(freq * (2*np.pi) * i / config['simu_freq'])) for i in range(N-N_ramp-Tc)]
#     target_force_3d[Tc+N_ramp:, 0] = [FX_MAX + 20.*(np.round(freq * (2*np.pi) * i / config['simu_freq'] - int(freq * (2*np.pi) * i / config['simu_freq']))-0.5) for i in range(N-N_ramp-Tc)]








N_START = int(config['T_CIRCLE'] * config['simu_freq']) 
print("N_start = ", N_START)

if(FILTER > 0):
    r1.data['contact_force_3d_measured'][:N] = analysis_utils.moving_average_filter(r1.data['contact_force_3d_measured'][:N].copy(), FILTER)
    r2.data['contact_force_3d_measured'][:N] = analysis_utils.moving_average_filter(r2.data['contact_force_3d_measured'][:N].copy(), FILTER) 
    # r3.data['contact_force_3d_measured'][:N] = analysis_utils.moving_average_filter(r3.data['contact_force_3d_measured'][:N].copy(), FILTER) 

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

plt.show()

# fig_f2, _ = s.plot_soft_contact_force([
#                            np.abs(r1.data['contact_force_3d_measured'][N_START:N] - target_force_3d[N_START:N]), 
#                            np.abs(r2.data['contact_force_3d_measured'][N_START:N] - target_force_3d[N_START:N]), 
#                           #  np.abs(r3.data['contact_force_3d_measured'][N_START:N] - target_force_3d[N_START:N]), 
#                            target_force_3d[N_START:N] - target_force_3d[N_START:N]
#                            ],
#                            [
#                             label1,
#                             label2, 
#                             # label3,
#                             'ref'
#                           ], 
#                           [
#                             'b', 
#                             'g', 
#                             # 'r', 
#                             'k'
#                             ],
#                           linestyle=[
#                             'solid', 
#                             'solid', 
#                             # 'solid', 
#                             'dotted'
#                             ])



# p_mea1 = get_p_(r1.data['joint_positions'][N_START:N,controlled_joint_ids][:,controlled_joint_ids], pinrobot.model, pinrobot.model.getFrameId('contact'))
# p_mea2 = get_p_(r2.data['joint_positions'][N_START:N,controlled_joint_ids][:,controlled_joint_ids], pinrobot.model, pinrobot.model.getFrameId('contact'))
# # p_mea3 = get_p_(r3.data['joint_positions'][N_START:N,controlled_joint_ids][:,controlled_joint_ids], pinrobot.model, pinrobot.model.getFrameId('contact'))
# target_position = np.zeros((N-N_START, 3))
# target_position[:,0] = r1.data['target_position_x'][N_START:N,0]
# target_position[:,1] = r1.data['target_position_y'][N_START:N,0]
# target_position[:,2] = r1.data['target_position_z'][N_START:N,0]
# target_position2 = np.zeros((N-N_START, 3))
# target_position2[:,0] = r2.data['target_position_x'][N_START:N,0]
# target_position2[:,1] = r2.data['target_position_y'][N_START:N,0]
# target_position2[:,2] = r2.data['target_position_z'][N_START:N,0]
# # target_position3 = np.zeros((N-N_START, 3))
# # target_position3[:,0] = r3.data['target_position_x'][N_START:N,0]
# # target_position3[:,1] = r3.data['target_position_y'][N_START:N,0]
# # target_position3[:,2] = r3.data['target_position_z'][N_START:N,0]
# fig_p1, _ = s.plot_ee_pos( [
#                            p_mea1,
#                            p_mea2 ,
#                           #  p_mea3,
#                            target_position 
#                            ],
#                            [
#                             label1, 
#                             label2, 
#                             # label3, 
#                             'ref'
#                           ], 
#                           [
#                             'b', 
#                             'g', 
#                             'r', 
#                             'k'
#                             ],
#                           linestyle=[
#                             'solid', 
#                             'solid', 
#                             'solid', 
#                             'dotted'
#                             ])

# fig_p2, _ = s.plot_ee_pos( [
#                            np.abs(p_mea1 - target_position),
#                            np.abs(p_mea2 - target_position2),
#                           #  np.abs(p_mea3 - target_position3),
#                            target_position - target_position
#                            ],
#                            [
#                             label1, 
#                             label2,
#                             # label3, 
#                             'ref'
#                           ], 
#                           [
#                             'b', 
#                             'g', 
#                             'r', 
#                             'k'
#                             ],
#                           linestyle=[
#                             'solid', 
#                             'solid', 
#                             'solid', 
#                             'dotted'
#                             ])


# # s.plot_joint_tau([r1.data['tau'][:,controlled_joint_ids] + r1.data['tau_gravity'][:,controlled_joint_ids],
# #                    r2.data['tau'][:,controlled_joint_ids] + r2.data['tau_gravity'][:,controlled_joint_ids],
# #                    r3.data['tau'][:,controlled_joint_ids] + r3.data['tau_gravity'][:,controlled_joint_ids]],
# #               ['Desired (+g(q))'+label1, 'Desired (+g(q))'+label2, 'Desired (+g(q))'+label3], 
# #               [[0.,0.,0.,0.], 'b', 'g', 'r'],
# #               ylims=[-model.effortLimit, +model.effortLimit] )



# fig, ax = plt.subplots(4, 1, sharex='col') 
# ax[0].plot(r1.data['time_df'], label='time_df'+label1)
# ax[0].plot(r2.data['time_df'], label='time_df'+label2)
# # ax[0].plot(r3.data['time_df'], label='time_df'+label3)

# ax[1].plot(r1.data['t_child'], label='child'+label1)
# ax[1].plot(r2.data['t_child'], label='child'+label2)
# # ax[1].plot(r3.data['t_child'], label='child'+label3)

# ax[2].plot(r1.data['ddp_iter'], label='iter'+label1)
# ax[2].plot(r2.data['ddp_iter'], label='iter'+label2)
# # ax[2].plot(r3.data['ddp_iter'], label='iter'+label3)

# ax[3].plot(r1.data['t_run'], label='t_run'+label1)
# ax[3].plot(r2.data['t_run'], label='t_run'+label2)
# # ax[3].plot(r3.data['t_run'], label='t_run'+label3)
# ax[0].legend(); ax[1].legend(); ax[2].legend(); ax[3].legend()


# print("------------------------------------")
# print("------------------------------------")
# print(label1+" Pxy error norm = ", np.linalg.norm(p_mea1[N_START:N,:2] - target_position[N_START:N,:2]) )
# print(label2+" Pxy error norm = ", np.linalg.norm(p_mea2[N_START:N,:2] - target_position2[N_START:N,:2]))
# # print(label3+" Pxy error norm = ", np.linalg.norm(p_mea3[N_START:N,:2] - target_position3[N_START:N,:2]))
# print(label1+" Fz mean abs error = ", np.mean(np.abs(r1.data['contact_force_3d_measured'][N_START:N, 2] - target_force_3d[N_START:N,2])))
# print(label2+" Fz mean abs error = ", np.mean(np.abs(r2.data['contact_force_3d_measured'][N_START:N, 2] - target_force_3d[N_START:N,2])))
# # print(label3+" Fz mean abs error = ", np.mean(np.abs(r3.data['contact_force_3d_measured'][N_START:N, 2] - target_force_3d[N_START:N,2])))
# print("------------------------------------")
# print("------------------------------------")


# # if(SAVE):
# #   fig_f1.savefig(data_path+label1+'_vs_'+label2+'_vs_'+label3+'_compare_force.png')
# #   fig_f2.savefig(data_path+label1+'_vs_'+label2+'_vs_'+label3+'_compare_force_err.png')
# #   # fig_p1.savefig(data_path+label1+'_vs_'+label2+'_vs_'+label3'_compare_pos.png')
# #   fig_p2.savefig(data_path+label1+'_vs_'+label2+'_vs_'+label3+'_compare_pos_err.png')


# plt.show()