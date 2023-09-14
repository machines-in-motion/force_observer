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


FILTER = False

PLOT = "SOTA"
# PLOT = "LATERAL_MODE"
# PLOT = "DF_DTAU"

color_list = ['b', 'g', 'r', 'y']

if PLOT == "SOTA":
    omega = 3
    print("Load data 1...")
    data_path = '/home/skleff/Desktop/delta_f_real_exp/sanding/lat_model/'
    r1 = DataReader(data_path+'config_REAL_2023-09-13T19:41:20.484038_medium_default.mds') 
    # r1 = DataReader("/home/skleff/Desktop/delta_f_real_exp/sanding/d_tau_vs_df/"+'config_REAL_2023-09-14T13:30:54.773869_ff_FI.mds') 
    print("Load data 2...")
    r2 = DataReader(data_path+'config_REAL_2023-09-13T19:44:20.481230_medium_ff.mds')  
    print("Load data 3...")
    data_path = "/home/skleff/Desktop/delta_f_real_exp/sanding/d_tau_vs_df/"
    r3 = DataReader(data_path+'config_REAL_2023-09-14T13:34:53.546969_ff_df_int.mds') 

    label1 = 'Default'
    # label1 = 'FI'
    label2 = 'Feedforward'
    label3 = 'Feedforward + Estimation'




if PLOT == "LATERAL_MODE":
    # TASK = "slow"
    TASK = "medium"
    # TASK = "fast"


    data_path = '/home/skleff/Desktop/delta_f_real_exp/sanding/lat_model/'

    if TASK == "slow":    
        print("Load data 1...")
        r1 = DataReader(data_path+'config_REAL_2023-09-13T19:30:00.596928_slow_default.mds') 
        print("Load data 2...")
        r2 = DataReader(data_path+'config_REAL_2023-09-13T19:20:23.053442_slow_coulomb.mds')  
        print("Load data 3...")
        r3 = DataReader(data_path+'config_REAL_2023-09-13T19:22:33.925757_slow_ff.mds') 

        omega = 1

    if TASK == "medium":    
        print("Load data 1...")
        r1 = DataReader(data_path+'config_REAL_2023-09-13T19:41:20.484038_medium_default.mds') 
        print("Load data 2...")
        r2 = DataReader(data_path+'config_REAL_2023-09-13T19:42:59.978248_medium_coulomb.mds')  
        print("Load data 3...")
        r3 = DataReader(data_path+'config_REAL_2023-09-13T19:44:20.481230_medium_ff.mds') 

        omega = 3

    if TASK == "fast":    
        print("Load data 1...")
        r1 = DataReader(data_path+'config_REAL_2023-09-13T18:52:30.333938_fast_default.mds') 
        print("Load data 2...")
        r2 = DataReader(data_path+'config_REAL_2023-09-13T18:55:15.619068_fast_coulomb.mds')  
        print("Load data 3...")
        r3 = DataReader(data_path+'config_REAL_2023-09-13T18:53:34.099135_fast_ff.mds') 

        omega = 6


    label1 = 'Default'
    label2 = 'Coulomb'
    label3 = 'Feedforward'

if PLOT == "DF_DTAU":
    omega = 3
    
    data_path = "/home/skleff/Desktop/delta_f_real_exp/sanding/d_tau_vs_df/"
    print("Load data 1...")
    r1 = DataReader(data_path+'config_REAL_2023-09-14T13:29:38.261347_ff_dtau_int.mds') 
    print("Load data 2...")
    r2 = DataReader(data_path+'config_REAL_2023-09-14T13:28:17.149401_ff_dtau_ext.mds') 
    print("Load data 3...")
    r3 = DataReader(data_path+'config_REAL_2023-09-14T13:34:53.546969_ff_df_int.mds') 
    print("Load data 4...")
    r4 = DataReader(data_path+'config_REAL_2023-09-14T13:26:58.523294_ff_df_ext.mds') 

    label1 = 'Torque Estimation (Model)'
    label2 = 'Torque Estimation (Feedforward)'
    label3 = 'Force Estimation (Model)'
    label4 = 'Force Estimation (Feedforward)'
 
    color_list = ['b', 'g', 'y', 'r']



FILTER = 400
N_START = 7000 
SKIP = 2000
    
from core_mpc import analysis_utils 

N = min(r1.data['tau'].shape[0], r2.data['tau'].shape[0])
N = min(N, r3.data['tau'].shape[0])

 
print("N_start = ", N_START)




force_3d_1 = np.zeros((N-N_START, 3))
force_3d_2 = np.zeros((N-N_START, 3))
force_3d_3 = np.zeros((N-N_START, 3))
force_3d_4 = np.zeros((N-N_START, 3))



if(FILTER > 0):
    force_3d_1 = analysis_utils.moving_average_filter(r1.data['contact_force_3d_measured'][N_START:N].copy(), FILTER)
    force_3d_2 = analysis_utils.moving_average_filter(r2.data['contact_force_3d_measured'][N_START:N].copy(), FILTER) 
    force_3d_3 = analysis_utils.moving_average_filter(r3.data['contact_force_3d_measured'][N_START:N].copy(), FILTER) 
    if PLOT == "DF_DTAU":
        force_3d_4 = analysis_utils.moving_average_filter(r4.data['contact_force_3d_measured'][N_START:N].copy(), FILTER) 
else:
    force_3d_1 = r1.data['contact_force_3d_measured'][N_START:N]
    force_3d_2 = r2.data['contact_force_3d_measured'][N_START:N] 
    force_3d_3 = r3.data['contact_force_3d_measured'][N_START:N] 
    if PLOT == "DF_DTAU":
        force_3d_4 = r4.data['contact_force_3d_measured'][N_START:N] 

target_force_3d = np.zeros((N, 3))
Tc = int(config['T_CONTACT'] * config['simu_freq'])
target_force_3d[Tc:, :3] = np.asarray(config['frameForceRef'])[:3]
N_ramp = int((config['T_RAMP'] - config['T_CONTACT']) * config['simu_freq'])           
FZ_MIN = 5. 
FZ_MAX = 50.
FX_MAX = config['frameForceRef'][0]
target_force_3d[Tc:Tc+N_ramp, 2] = [FZ_MIN + (FZ_MAX - FZ_MIN)*i/N_ramp for i in range(N_ramp)]
target_force_3d[Tc+N_ramp:, 2] = FZ_MAX

time_lin = np.linspace(0, (N-N_START-SKIP) / config['simu_freq'], (N-N_START-SKIP))


# Generate plot of number of iterations for each problem
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
 
    
    
fig0, ax = plt.subplots(3, 1, figsize=(10.8,10.8))

for i in range(3):
    ax[i].plot(time_lin, target_force_3d[N_START+SKIP:N,i], color='k', linewidth=4, linestyle='--', label='Reference', alpha=0.5) 
    ax[i].plot(time_lin, force_3d_1[SKIP:,i], color='b', linewidth=4, label=label1, alpha=1.)
    ax[i].plot(time_lin, force_3d_2[SKIP:,i], color='g', linewidth=4, label=label2, alpha=1.)
    ax[i].plot(time_lin, force_3d_3[SKIP:,i], color='r', linewidth=4, label=label3, alpha=1.)
    if PLOT == "DF_DTAU":
        ax[i].plot(time_lin, force_3d_4[SKIP:,i], color='r', linewidth=4, label=label4, alpha=1.)
    
    
    ax[i].grid(True) 
ax[0].legend() 
ax[0].set_ylabel('F (N)', fontsize=26)
ax[0].set_xlabel('Time (s)', fontsize=26)
ax[0].tick_params(axis = 'y', labelsize=22)
ax[0].tick_params(axis = 'x', labelsize=22)


fig1 = plt.figure(figsize=(20., 10.))

plt.plot(time_lin, target_force_3d[N_START+SKIP:N, 2], color='k', linewidth=4, linestyle='--', label='Reference', alpha=0.5) 
plt.plot(time_lin, force_3d_1[SKIP:,2], color=color_list[0], linewidth=4, label=label1, alpha=0.8)
plt.plot(time_lin, force_3d_2[SKIP:,2], color=color_list[1], linewidth=4, label=label2, alpha=0.8)
plt.plot(time_lin, force_3d_3[SKIP:,2], color=color_list[2], linewidth=4, label=label3, alpha=0.8)
if PLOT == "DF_DTAU":
    plt.plot(time_lin, force_3d_4[SKIP:,2], color=color_list[3], linewidth=4, label=label4, alpha=1.)
        
plt.grid(True) 
# plt.legend(loc='upper right', framealpha=0.95, fontsize=26) 
plt.legend(framealpha=0.95, fontsize=26) 
plt.xlim(time_lin[0], time_lin[-1])
plt.ylabel('F (N)', fontsize=26)
plt.xlabel('Time (s)', fontsize=26)
plt.tick_params(axis = 'y', labelsize=22)
plt.tick_params(axis = 'x', labelsize=22)


   

if PLOT == "SOTA":
    fig1.savefig('/home/skleff/Desktop/delta_f_real_exp/sanding/' + PLOT + '.pdf', bbox_inches="tight")
if PLOT == "LATERAL_MODE":
    fig1.savefig('/home/skleff/Desktop/delta_f_real_exp/sanding/lat_model/' + TASK + '.pdf', bbox_inches="tight")
if PLOT == "DF_DTAU":
    fig1.savefig('/home/skleff/Desktop/delta_f_real_exp/sanding/d_tau_vs_df/' + PLOT + '.pdf', bbox_inches="tight")



# Compute average tracking error for each circle
CIRCLE_PERIOD_IN_CYCLES = int(2*np.pi/omega*1000)
print("CIRCLE PERIOD = ", CIRCLE_PERIOD_IN_CYCLES)

N_CIRCLE = int((N-N_START)/CIRCLE_PERIOD_IN_CYCLES)
N = N_START + CIRCLE_PERIOD_IN_CYCLES * N_CIRCLE

# assert N_CIRCLE >= 11
N_CIRCLE = 11
print("N_CIRCLE (fisrt is desregarded) = ", N_CIRCLE - 1)


def print_error(r, label):
    error_traj = np.abs(r.data['contact_force_3d_measured'][N_START:, 2] - target_force_3d[N_START:, 2])
    mean_errors = [np.mean(error_traj[t*CIRCLE_PERIOD_IN_CYCLES:(t+1)*CIRCLE_PERIOD_IN_CYCLES]) for t in range(1, N_CIRCLE)]
    # print(mean_errors)
    print(label, ": F mean abs error      = ",  np.mean(mean_errors), "  +-  ", np.std(mean_errors))
    print('\n')
    
    
    
    
print_error(r1, label1)
print_error(r2, label2)
print_error(r3, label3)
if PLOT == "DF_DTAU":
    print_error(r4, label4)



plt.show()
plt.close('all')