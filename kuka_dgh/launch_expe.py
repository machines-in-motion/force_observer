import pybullet as p
import numpy as np
from dynamic_graph_head import ThreadHead, SimHead, HoldPDController, SimForcePlate
import pinocchio as pin 

from datetime import datetime 

import dynamic_graph_manager_cpp_bindings
from robot_properties_kuka.config import IiwaReducedConfig
from robot_properties_kuka.iiwaReducedWrapper import IiwaReducedRobot
from bullet_utils.env import BulletEnvWithGround

# from utils.find_contact_point import compute_sensor_frame_transform
import sys
sys.path.append("../../")
from force_feedback_dgh.demos.utils.find_contact_point import compute_sensor_frame_transform



from controller import ClassicalMPCContact
# from controller36d import ClassicalMPCContact
from core_mpc import path_utils, sim_utils
from core_mpc.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

SIM = False

DGM_PARAMS_PATH = "/home/skleff/ws/workspace/install/robot_properties_kuka/lib/python3.8/site-packages/robot_properties_kuka/robot_properties_kuka/dynamic_graph_manager/dgm_parameters_iiwa.yaml"
CONFIG_NAME = 'config' 
# CONFIG_NAME = 'config36d'
CONFIG_PATH = CONFIG_NAME+".yml"


CONTROLLED_JOINTS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
QREF              = np.zeros(7)
# Load robot model, config and contact info
pin_robot                     = IiwaReducedConfig.buildRobotWrapper(controlled_joints=CONTROLLED_JOINTS, qref=QREF)
config                        = path_utils.load_yaml_file(CONFIG_PATH)
id_endeff                     = pin_robot.model.getFrameId(config['frame_of_interest'])
contactTranslationTarget      = np.asarray(config['contactPosition']) + np.asarray(config['oPc_offset'])
contact_placement             = pin.SE3(np.eye(3), np.asarray(config['contactPosition'])) 
oPc                           = contact_placement.translation + np.asarray(config['oPc_offset'])
if('1D' in config['contactType']):
    logger.warning("Contact type 1D detected")
elif(config['contactType'] == '3D'):
    logger.warning("Contact type 3D detected")
else:
    logger.error("Unknown contact type !!!")

# Add FT sensor frame to the model of the robot 
# Placement is determined using script in demos/utils/find_contact_point.py
_, _, cMs, _ = compute_sensor_frame_transform(pin_robot, IiwaReducedConfig.cad_origin_name)


if SIM:
    config['T_tot'] = 400
    # Sim env + set initial state 
    env = BulletEnvWithGround(p.DIRECT)
    robot_simulator = env.add_robot(IiwaReducedRobot(controlled_joints=CONTROLLED_JOINTS, qref=QREF))
    robot_simulator.pin_robot = pin_robot 
    q_init = np.asarray(config['q0'] )
    v_init = np.asarray(config['dq0'])
    robot_simulator.reset_state(q_init, v_init)
    robot_simulator.forward_robot(q_init, v_init)
    # Display reach target, contact surface & desired contact point 
    sim_utils.display_ball(contactTranslationTarget, RADIUS=0.02, COLOR=[1.,0.,0.,0.2])
    contactId = sim_utils.display_contact_surface(contact_placement, bullet_endeff_ids=robot_simulator.bullet_endeff_ids)
    sim_utils.set_lateral_friction(contactId, 0.5)
    sim_utils.set_contact_stiffness_and_damping(contactId, 10000, 500)
    sim_utils.display_ball(oPc, robot_simulator.pin_robot.data.oMf[robot_simulator.pin_robot.model.getFrameId('iiwa_base')], RADIUS=0.01, COLOR=[1.,0.,0.,1.])
    # Get initial force from simulator 
    f0 = np.zeros(6) #sim_utils.get_contact_wrench(robot_simulator, id_endeff, config['pinRefFrame'])  
    head = SimHead(robot_simulator, with_sliders=False, with_force_plate=True)
else:
    config['T_tot'] = 400
    # Get initial force from sensor reading (should be 0)
    f0 = np.zeros(6) #sim_utils.get_contact_wrench(robot_simulator, id_endeff, softContactModel.pinRefFrame)  
    path = DGM_PARAMS_PATH 
    head = dynamic_graph_manager_cpp_bindings.DGMHead(path)
    target = None
    env = None

ctrl = ClassicalMPCContact(head, pin_robot, config, f0, contact_placement, cMs, run_sim=SIM, controlled_joints_names=CONTROLLED_JOINTS)



if(SIM):
    thread_head = ThreadHead(
        1./config['simu_freq'],                                         # dt.
        HoldPDController(head, 50., 0.5, with_sliders=False),           # Safety controllers.
        head,                                                           # Heads to read / write from.
        [('force_plate', SimForcePlate([robot_simulator]))],            # Simulated force plate
        env                                                             # Environment to step.
    )
else:
    thread_head = ThreadHead(
        1./config['simu_freq'],                                         # dt.
        HoldPDController(head, 50., 0.5, with_sliders=False),           # Safety controllers.
        head,                                                           # Heads to read / write from.
        [],
        env                                                             # Environment to step.
    )



thread_head.switch_controllers(ctrl)

prefix = "/home/skleff/Desktop/delta_f_real_exp/video/"
suffix = "_DEFAULT"


if SIM:
    thread_head.start_logging(20, prefix+CONFIG_NAME+"_SIM_"+str(datetime.now().isoformat())+suffix+".mds")
    # thread_head.start_logging(int(config['T_tot']), prefix+CONFIG_NAME+"_SIM_"+str(datetime.now().isoformat())+suffix+".mds")
    thread_head.sim_run_timed(int(20*config['simu_freq']))
    thread_head.stop_logging()
    # thread_head.plot_timing()
else:
    thread_head.start()
    thread_head.start_logging(28, prefix+CONFIG_NAME+"_REAL_"+str(datetime.now().isoformat())+suffix+".mds")
    # thread_head.start_logging(30, prefix+CONFIG_NAME+"_REAL_"+str(datetime.now().isoformat())+suffix+".mds")
    
# reccord times for circles:
# Slow: 77s
# Medium: 32s
# Fast: 20s