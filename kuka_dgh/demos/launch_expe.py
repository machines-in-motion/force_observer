import pybullet as p
import numpy as np
from dynamic_graph_head import ThreadHead, SimHead, HoldPDController, SimForcePlate
from datetime import datetime
import dynamic_graph_manager_cpp_bindings
from mim_robots.robot_loader import load_bullet_wrapper, load_pinocchio_wrapper
from mim_robots.pybullet.env import BulletEnvWithGround
from mim_robots.robot_list import MiM_Robots
import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'kuka_dgh'
os.sys.path.insert(1, str(python_path))
print(python_path)
import launch_utils
from utils.find_contact_point import compute_sensor_frame_transform
from utils.sim_utils import display_ball, display_contact_surface, set_lateral_friction, set_contact_stiffness_and_damping
import pinocchio as pin






# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Choose experiment, load config and import controller  #  
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
SIM           = True
EXP_NAME      = 'polishing' # <<<<<<<<<<<<< Choose experiment here (cf. launch_utils)
config        = launch_utils.load_config_file(EXP_NAME)
MPCController = launch_utils.import_mpc_controller(EXP_NAME)
    
    



    
# # # # # # # # # # # #
# Import robot model  #
# # # # # # # # # # # #
LOCKED_JOINTS = ['A7']
pin_robot     = load_pinocchio_wrapper('iiwa_ft_sensor_shell', locked_joints=LOCKED_JOINTS)

# Add FT sensor frame to the model of the robot - this calibration routine changes the pinocchio model !
_, _, cMs, _ = compute_sensor_frame_transform(pin_robot, mount_piece_type='shell')



# # # # # # # # # # # # #
# Setup control thread  #
# # # # # # # # # # # # #
if SIM:
    # Sim env + set initial state 
    config['T_tot'] = 400              
    env = BulletEnvWithGround(p.GUI)
    robot_simulator = load_bullet_wrapper('iiwa_ft_sensor_shell', locked_joints=LOCKED_JOINTS)
    robot_simulator.pin_robot = pin_robot 
    env.add_robot(robot_simulator)
    q_init = np.asarray(config['q0'] )
    v_init = np.asarray(config['dq0'])
    robot_simulator.reset_state(q_init, v_init)
    robot_simulator.forward_robot(q_init, v_init)
    # <<<<< Customize your PyBullet environment here if necessary
    # Display reach target, contact surface & desired contact point 
    display_ball(np.asarray(config['contactPosition']) + np.asarray(config['oPc_offset']), RADIUS=0.02, COLOR=[1.,0.,0.,0.2])
    contactId = display_contact_surface(pin.SE3(np.eye(3), np.asarray(config['contactPosition'])), bullet_endeff_ids=robot_simulator.bullet_endeff_ids)
    set_lateral_friction(contactId, 0.5)
    set_contact_stiffness_and_damping(contactId, 10000, 500)
    head = SimHead(robot_simulator, with_sliders=False, with_force_plate=True)
# !!!!!!!!!!!!!!!!
# !! REAL ROBOT !!
# !!!!!!!!!!!!!!!!
else:
    config['T_tot'] = 400              
    path = MiM_Robots['iiwa'].dgm_path  
    print(path)
    head = dynamic_graph_manager_cpp_bindings.DGMHead(path)
    target = None
    env = None

ctrl = MPCController(head, pin_robot, config, run_sim=SIM, cMs=cMs, locked_joints=LOCKED_JOINTS)


if(SIM):
    thread_head = ThreadHead(
        1./config['ctrl_freq'],                                         # dt.
        HoldPDController(head, 50., 0.5, with_sliders=False),           # Safety controllers.
        head,                                                           # Heads to read / write from.
        [('force_plate', SimForcePlate([robot_simulator]))],            # Simulated force plate
        env                                                             # Environment to step.
    )
else:
    thread_head = ThreadHead(
        1./config['ctrl_freq'],                                         # dt.
        HoldPDController(head, 50., 0.5, with_sliders=False),           # Safety controllers.
        head,                                                           # Heads to read / write from.
        [],
        env                                                             # Environment to step.
    )

thread_head.switch_controllers(ctrl)




# # # # # # # # #
# Data logging  #
# # # # # # # # # <<<<<<<<<<<<< Choose data save path & log config here (cf. launch_utils)
# prefix     = "/home/skleff/data_sqp_paper_croc2/constrained/circle/"
prefix     = "/tmp/"
suffix     = 'test'
LOG_FIELDS = launch_utils.get_log_config(EXP_NAME) 





# # # # # # # # # # # 
# Launch experiment #
# # # # # # # # # # # 
if SIM:
    thread_head.start_logging(int(config['T_tot']), prefix+EXP_NAME+"_SIM_"+str(datetime.now().isoformat())+suffix+".mds", LOG_FIELDS=LOG_FIELDS)
    thread_head.sim_run_timed(int(config['T_tot']))
    thread_head.stop_logging()
else:
    thread_head.start()
    thread_head.start_logging(30, prefix+EXP_NAME+"_REAL_"+str(datetime.now().isoformat())+suffix+".mds", LOG_FIELDS=LOG_FIELDS)
    
thread_head.plot_timing() # <<<<<<<<<<<<< Comment out to skip timings plot
    
# reccord times for circles:
# Slow: 77s
# Medium: 32s
# Fast: 20s