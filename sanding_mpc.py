"""
@package force_observer
@file sanding_MPC.py
@author Sebastien Kleff & Armand Jordana
@license License BSD-3-Clause
@copyright Copyright (c) 2021, New York University & LAAS-CNRS
@date 2021-10-28
@brief Closed-loop classical MPC for sanding task
"""

import sys
sys.path.append('.')

from core_mpc.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


import numpy as np  
np.set_printoptions(precision=4, linewidth=180)
RANDOM_SEED = 19


from core_mpc import path_utils, pin_utils, mpc_utils, misc_utils
from core_mpc import ocp as ocp_utils
from core_mpc import sim_utils as simulator_utils

from classical_mpc.data import DDPDataHandlerClassical
from ocp_utils import OptimalControlProblemClassicalWithObserver
from mpc_utils import MPCDataHandlerClassicalWithEstimator
from estimator import Estimator


import time
import pinocchio as pin
jRc = np.eye(3)
jpc = np.array([0, 0., 0.12])
jMc = pin.SE3(jRc, jpc)

def solveOCP(q, v, ddp, nb_iter, node_id_reach, target_reach, node_id_contact, node_id_track, node_id_circle, force_weight, TASK_PHASE, target_force):
        t = time.time()
        x = np.concatenate([q, v])
        ddp.problem.x0 = x
        xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
        xs_init[0] = x
        us_init = list(ddp.us[1:]) + [ddp.us[-1]] 
        # Get OCP nodes
        m = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
        # Update OCP for reaching phase
        if(TASK_PHASE == 1):
            # If node id is valid
            if(node_id_reach <= ddp.problem.T and node_id_reach >= 0):
                # Updates nodes between node_id and terminal node 
                for k in range( node_id_reach, ddp.problem.T+1, 1 ):
                    m[k].differential.costs.costs["translation"].active = True
                    m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
        # Update OCP for contact phase
        if(TASK_PHASE == 3):
            # If node id is valid
            if(node_id_contact <= ddp.problem.T and node_id_contact >= 0):
                # Updates nodes between node_id and terminal node 
                for k in range( node_id_contact, ddp.problem.T+1, 1 ):
                    # wf = min(1.*(k + 1. - node_id_contact) , force_weight)  
                    m[k].differential.costs.costs["translation"].active = True
                    m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                    m[k].differential.costs.costs["translation"].cost.activation.weights = np.array([1., 1., 0.])
                    m[k].differential.costs.costs["velocity"].active = True
                    # activate contact and force cost
                    m[k].differential.contacts.changeContactStatus("contact", True)
                    if(k < ddp.problem.T):
                        fref = pin.Force(np.array([0., 0., target_force[k], 0., 0., 0.]))
                        m[k].differential.costs.costs["force"].active = True
                        m[k].differential.costs.costs["force"].cost.residual.reference = fref
        # Update OCP for circle phase
        if(TASK_PHASE == 4):
            # If node id is valid
            if(node_id_circle <= ddp.problem.T and node_id_circle >= 0):
                # Updates nodes between node_id and terminal node
                for k in range( node_id_circle, ddp.problem.T+1, 1 ):
                    m[k].differential.costs.costs["translation"].active = True
                    m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                    m[k].differential.costs.costs["translation"].cost.activation.weights = np.array([1., 1., 0.])
                    m[k].differential.costs.costs["translation"].weight = 10. # 10000
                    m[k].differential.costs.costs["velocity"].active = False
                    # activate contact and force cost
                    m[k].differential.contacts.changeContactStatus("contact", True)
                    if(k < ddp.problem.T):
                        fref = pin.Force(np.array([0., 0., target_force[k], 0., 0., 0.]))
                        m[k].differential.costs.costs["force"].active = True
                        m[k].differential.costs.costs["force"].weight = 1000 # 1000
                        m[k].differential.costs.costs["force"].cost.residual.reference = fref
        # get predicted force from rigid model (careful : expressed in LOCAL !!!)
        j_wrenchpred = ddp.problem.runningDatas[0].differential.multibody.contacts.contacts['contact'].f
        fpred = jMc.actInv(j_wrenchpred).linear
        # print(fpred)
        problem_formulation_time = time.time()
        t_child_1 =  problem_formulation_time - t
        # Solve OCP 
        ddp.solve(xs_init, us_init, maxiter=nb_iter, isFeasible=False)
        # Send solution to parent process + riccati gains
        solve_time = time.time()
        ddp_iter = ddp.iter
        t_child =  solve_time - problem_formulation_time
        return ddp.us, ddp.xs, ddp.K, t_child, ddp_iter, t_child_1, fpred



PLOT_INIT=False


# # # # # # # # # # # # # # # # # # #
### LOAD ROBOT MODEL and SIMU ENV ### 
# # # # # # # # # # # # # # # # # # # 

# Read config file, create a simulation environment & simu-pin wrapper 
config = path_utils.load_yaml_file('/home/skleff/misc_repos/force_observer/sanding_mpc.yml')
dt_simu = 1./float(config['simu_freq'])  
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
env, robot_simulator, _ = simulator_utils.init_bullet_simulation('iiwa', dt=dt_simu, x0=x0)
robot = robot_simulator.pin_robot
# Get dimensions & frame of interest
nq, nv = robot.model.nq, robot.model.nv; nu = nq
frame_of_interest = config['frame_of_interest']
id_endeff = robot.model.getFrameId(frame_of_interest)

# EE translation target : contact point + vertical offset (radius of the ee ball)
contactTranslationTarget = np.asarray(config['contactPosition']) + np.asarray(config['oPc_offset'])
simulator_utils.display_ball(contactTranslationTarget, RADIUS=0.02, COLOR=[1.,0.,0.,0.2])
# Display contact surface + set contact simulation parameters
contact_placement        = pin.SE3(np.eye(3), np.asarray(config['contactPosition']))
contact_surface_bulletId = simulator_utils.display_contact_surface(contact_placement, bullet_endeff_ids=robot_simulator.bullet_endeff_ids, TILT=[config['TILT_PITCH_LOCAL_DEG']*np.pi/180, 0., 0.])
simulator_utils.set_lateral_friction(contact_surface_bulletId, 0.5)
simulator_utils.set_contact_stiffness_and_damping(contact_surface_bulletId, 1e6, 1e3)



# # # # # # # # # 
### OCP SETUP ###
# # # # # # # # # 

# Init shooting problem and solver
delta_f = np.zeros(3)
ddp = OptimalControlProblemClassicalWithObserver(robot, config).initialize(x0, delta_f, callbacks=False)
# !!! Deactivate all costs & contact models initially !!!
models = list(ddp.problem.runningModels) + [ddp.problem.terminalModel]
for k,m in enumerate(models):
    m.differential.costs.costs["translation"].active = False
    if(k < config['N_h']):
          m.differential.costs.costs["force"].active = False
          m.differential.costs.costs["force"].cost.residual.reference = pin.Force.Zero()
    m.differential.contacts.changeContactStatus("contact", False)
# Warmstart and solve
xs_init = [x0 for i in range(config['N_h']+1)]
us_init = [pin_utils.get_u_grav(q0, robot.model, np.zeros(nv)) for i in range(config['N_h'])] 
ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)

# Plot initial solution
if(PLOT_INIT):
  ddp_handler = DDPDataHandlerClassical(ddp)
  ddp_data = ddp_handler.extract_data(frame_of_interest, frame_of_interest)
  _, _ = ddp_handler.plot_ddp_results(ddp_data, markers=['.'], SHOW=True)



# Create estimator 
force_estimator = Estimator(robot_simulator.pin_robot, 1, id_endeff, config['contacts'][0]['contactModelGains'])


# Setup tracking problem with circle ref EE trajectory + Warm start state = IK of circle trajectory
RADIUS = config['frameCircleTrajectoryRadius'] 
OMEGA  = config['frameCircleTrajectoryVelocity']
# Force trajectory
F_MIN = 0.
F_MAX = config['frameForceRef'][2]
N_total = int((config['T_tot'] - config['T_CONTACT'])/config['dt'] + config['N_h'])
N_min  = 5
N_ramp = N_min + 10
target_force_traj = np.zeros( (N_total, 3) )
target_force_traj[0:N_min*config['N_h'], 2] = F_MIN
target_force_traj[N_min*config['N_h']:N_ramp*config['N_h'], 2] = [F_MIN + (F_MAX - F_MIN)*i/((N_ramp-N_min)*config['N_h']) for i in range((N_ramp-N_min)*config['N_h'])]
target_force_traj[N_ramp*config['N_h']:, 2] = F_MAX
target_force = np.zeros(config['N_h']+1)
force_weight = np.asarray(config['frameForceWeight'])
# Circle trajectory 
N_total_pos = int((config['T_tot'] - config['T_REACH'])/config['dt'] + config['N_h'])
N_circle = int((config['T_tot'] - config['T_CIRCLE'])/config['dt']) + config['N_h']
target_position_traj = np.zeros( (N_total_pos, 3) )
# absolute desired position
pdes = np.asarray(config['contactPosition']) + np.asarray(config['oPc_offset'])
target_position_traj[0:N_circle, :] = [np.array([pdes[0] + RADIUS * np.sin(i*config['dt']*OMEGA), 
                                                 pdes[1] + RADIUS * (1-np.cos(i*config['dt']*OMEGA)),
                                                 pdes[2]]) for i in range(N_circle)]
target_position_traj[N_circle:, :] = target_position_traj[N_circle-1,:]
target_position = np.zeros((config['N_h']+1, 3)) 
target_position[:,:] = pdes.copy()









# # # # # # # # # # #
### INIT MPC SIMU ###
# # # # # # # # # # #
sim_data = MPCDataHandlerClassicalWithEstimator(config, robot)
sim_data.init_sim_data(x0)
sim_data.record_simu_cycle_measured(0, x0, x0, us_init[0], np.zeros(6))

  # Replan & control counters
nb_plan = 0
nb_ctrl = 0
# Additional simulation blocks 
communicationModel = mpc_utils.CommunicationModel(config)
actuationModel     = mpc_utils.ActuationModel(config, nu=nu, SEED=RANDOM_SEED)
sensingModel       = mpc_utils.SensorModel(config, SEED=RANDOM_SEED)
torqueController   = mpc_utils.LowLevelTorqueController(config, nu=nu)
antiAliasingFilter = mpc_utils.AntiAliasingFilter()

# Display target circle  trajectory (reference)
nb_points = 20 
for i in range(nb_points):
  t = (i/nb_points)*2*np.pi/OMEGA
  pos = ocp_utils.circle_point_WORLD(t, contact_placement, radius=RADIUS, omega=OMEGA, LOCAL_PLANE=config['CIRCLE_LOCAL_PLANE'])
  simulator_utils.display_ball(pos, RADIUS=0.01, COLOR=[1., 0., 0., 1.])

draw_rate = 1000

# # # # # # # # # # # #
### SIMULATION LOOP ###
# # # # # # # # # # # #

# Horizon in simu cycles
node_id_reach   = -1
node_id_contact = -1
node_id_track   = -1
node_id_circle  = -1
TASK_PHASE      = 0
NH_SIMU   = int(config['N_h']*sim_data.dt/sim_data.dt_simu)
T_REACH   = int(config['T_REACH']/sim_data.dt_simu)
T_CONTACT = int(config['T_CONTACT']/sim_data.dt_simu)
T_CIRCLE   = int(config['T_CIRCLE']/sim_data.dt_simu)
OCP_TO_MPC_CYCLES = 1./(sim_data.dt_plan / config['dt'])
OCP_TO_SIMU_CYCLES = 1./(sim_data.dt_simu / config['dt'])
logger.debug("Size of MPC horizon in simu cycles     = "+str(NH_SIMU))
logger.debug("Start of reaching phase in simu cycles = "+str(T_REACH))
logger.debug("Start of contact phase in simu cycles  = "+str(T_CONTACT))
logger.debug("Start of circle phase in simu cycles   = "+str(T_CIRCLE))
logger.debug("OCP to PLAN time ratio = "+str(OCP_TO_MPC_CYCLES))


# SIMULATE
sim_data.tau_mea_SIMU[0,:] = ddp.us[0]
err_fz = 0
err_p = 0
count=0

for i in range(sim_data.N_simu): 

    if(i%config['log_rate']==0 and config['LOG']): 
      print('')
      logger.info("SIMU step "+str(i)+"/"+str(sim_data.N_simu))
      print('')


    # # # # # # # # # 
    # # Update OCP  #
    # # # # # # # # # 
    time_to_reach   = int(i - T_REACH)
    time_to_contact = int(i - T_CONTACT)
    time_to_circle  = int(i - T_CIRCLE)

    if(time_to_reach == 0): 
        logger.warning("Entering reaching phase")
    # If tracking phase enters the MPC horizon, start updating models from the end with tracking models      
    if(0 <= time_to_reach and time_to_reach <= NH_SIMU):
        TASK_PHASE = 1
        # If current time matches an OCP node 
        if(time_to_reach%OCP_TO_SIMU_CYCLES == 0):
            # Select IAM
            node_id_reach = config['N_h'] - int(time_to_reach/OCP_TO_SIMU_CYCLES)


    if(time_to_contact == 0): 
        # Record end-effector position at the time of the contact switch
        position_at_contact_switch = robot.data.oMf[id_endeff].translation.copy()
        target_position[:,:] = position_at_contact_switch.copy()
        logger.warning("Entering contact phase")
    # If contact phase enters horizon start updating models from the the end with contact models
    if(0 <= time_to_contact and time_to_contact <= NH_SIMU):
        TASK_PHASE = 3
        # If current time matches an OCP node 
        if(time_to_contact%OCP_TO_SIMU_CYCLES == 0):
            # Select IAM
            node_id_contact = config['N_h'] - int(time_to_contact/OCP_TO_SIMU_CYCLES)

    if(0 <= time_to_contact and time_to_contact%OCP_TO_SIMU_CYCLES == 0):
        ti  = int(time_to_contact/OCP_TO_SIMU_CYCLES)
        tf  = ti + config['N_h']+1
        target_force = target_force_traj[ti:tf,2]


    if(time_to_circle == 0): 
        logger.warning("Entering circle phase")
    # If circle tracking phase enters the MPC horizon, start updating models from the end with tracking models      
    if(0 <= time_to_circle and time_to_circle <= NH_SIMU):
        TASK_PHASE = 4
        # If current time matches an OCP node 
        if(time_to_circle%OCP_TO_SIMU_CYCLES == 0):
            # Select IAM
            node_id_circle = config['N_h'] - int(time_to_circle/OCP_TO_SIMU_CYCLES)

    if(0 <= time_to_circle and time_to_circle%OCP_TO_SIMU_CYCLES == 0):
        # set position refs over current horizon
        ti  = int(time_to_circle/OCP_TO_SIMU_CYCLES)
        tf  = ti + config['N_h']+1
        # Target in (x,y)  = circle trajectory + offset to start from current position instead of absolute target
        offset_xy = position_at_contact_switch[:2] - pdes[:2]
        target_position[:,:2] = target_position_traj[ti:tf,:2] + offset_xy
        # Target in z is fixed to the anchor at switch (equals absolute target if RESET_ANCHOR = False)
        # No position tracking in z : redundant with zero activation weight on z
        target_position[:,2]  = robot_simulator.pin_robot.data.oMf[id_endeff].translation[2].copy()



    # # # # # # # # #
    # # Solve OCP # #
    # # # # # # # # #
    # Solve OCP if we are in a planning cycle (MPC/planning frequency)
    if(i%int(sim_data.simu_freq/sim_data.plan_freq) == 0):    
        # Anti-aliasing filter for measured state
        x_filtered = antiAliasingFilter.step(nb_plan, i, sim_data.plan_freq, sim_data.simu_freq, sim_data.state_mea_SIMU)
        # Reset x0 to measured state + warm-start solution
        q = x_filtered[:nq]
        v = x_filtered[nq:nq+nv]
        # Solve OCP 
        solveOCP(q, v, ddp, config['maxiter'], node_id_reach, target_position, node_id_contact, node_id_track, node_id_circle, force_weight, TASK_PHASE, target_force)
        # Record MPC predictions, cost references and solver data 
        sim_data.record_predictions(nb_plan, ddp)
        sim_data.record_cost_references(nb_plan, ddp)
        sim_data.record_solver_data(nb_plan, ddp) 
        # Model communication delay between computer & robot (buffered OCP solution)
        communicationModel.step(sim_data.x_pred, sim_data.u_curr)
        # Record interpolated desired state, control and force at MPC frequency
        sim_data.record_plan_cycle_desired(nb_plan)
        # Increment planning counter
        nb_plan += 1


    # # # # # # # # # #
    # # Send policy # #
    # # # # # # # # # #
    # If we are in a control cycle send reference torque to motor driver and compute the motor torque
    if(i%int(sim_data.simu_freq/sim_data.ctrl_freq) == 0):   
        # Anti-aliasing filter on measured torques (sim-->ctrl)
        tau_mea_CTRL            = antiAliasingFilter.step(nb_ctrl, i, sim_data.ctrl_freq, sim_data.simu_freq, sim_data.tau_mea_SIMU)
        tau_mea_derivative_CTRL = antiAliasingFilter.step(nb_ctrl, i, sim_data.ctrl_freq, sim_data.simu_freq, sim_data.tau_mea_derivative_SIMU)
        # Select the desired torque 
        tau_des_CTRL = sim_data.u_curr.copy()
        # Optionally interpolate to the control frequency using Riccati gains
        if(config['RICCATI']):
          x_filtered = antiAliasingFilter.step(nb_ctrl, i, sim_data.ctrl_freq, sim_data.simu_freq, sim_data.state_mea_SIMU)
          tau_des_CTRL += ddp.K[0].dot(ddp.problem.x0 - x_filtered)
        # Compute the motor torque 
        tau_mot_CTRL = torqueController.step(tau_des_CTRL, tau_mea_CTRL, tau_mea_derivative_CTRL)
        # Increment control counter
        nb_ctrl += 1


    # Simulate actuation 
    tau_mea_SIMU = actuationModel.step(i, tau_mot_CTRL, joint_vel=sim_data.state_mea_SIMU[i,nq:nq+nv])
    # Step PyBullet simulator
    robot_simulator.send_joint_command(tau_mea_SIMU)
    env.step()
    # Measure new state + forces from simulation 
    q_mea_SIMU, v_mea_SIMU = robot_simulator.get_state()
    robot_simulator.forward_robot(q_mea_SIMU, v_mea_SIMU)

    # !!! PyBullet forces come in LWA by default : transform to LOCAL 
    f_mea_SIMU_world = robot_simulator.end_effector_forces(sim_data.PIN_REF_FRAME)[1][0]
    if(sim_data.PIN_REF_FRAME == pin.LOCAL):
        lwaMc = robot_simulator.pin_robot.data.oMf[id_endeff]
        f_mea_SIMU = lwaMc.actInv(pin.Force(f_mea_SIMU_world)).vector
    fz_mea_SIMU = np.array([f_mea_SIMU[2]])
    if(i%1000==0): 
      logger.info("f_mea  = "+str(f_mea_SIMU))      
    # Record data (unnoised)
    x_mea_SIMU = np.concatenate([q_mea_SIMU, v_mea_SIMU]).T 
    # Simulate sensing 
    x_mea_noise_SIMU = sensingModel.step(x_mea_SIMU)
    # Record measurements of state, torque and forces 
    sim_data.record_simu_cycle_measured(i, x_mea_SIMU, x_mea_noise_SIMU, tau_mea_SIMU, f_mea_SIMU)


    # Estimation
    if(i>0): a_mea_SIMU = (v_mea_SIMU - sim_data.state_mea_SIMU[i-1, nv:]) / env.dt
    else: a_mea_SIMU = np.zeros(nv)
    if(np.linalg.norm(fz_mea_SIMU) > 1e-6):
        F, delta_f = force_estimator.estimate(q_mea_SIMU, v_mea_SIMU, a_mea_SIMU, tau_mea_SIMU, delta_f, fz_mea_SIMU)
    sim_data.record_simu_cycle_estimates(i, delta_f)

    # Display real 
    if(i%draw_rate==0):
      pos = robot_simulator.pin_robot.data.oMf[id_endeff].translation.copy()
      simulator_utils.display_ball(pos, RADIUS=0.01, COLOR=[0.,0.,1.,0.3])
    
# # # # # # # # # # #
# PLOT SIM RESULTS  #
# # # # # # # # # # #
save_dir = '/tmp'
save_name = 'kuka_sanding_estimator'+\
                        '_BIAS='+str(config['SCALE_TORQUES'])+\
                        '_NOISE='+str(config['NOISE_STATE'] or config['NOISE_TORQUES'])+\
                        '_DELAY='+str(config['DELAY_OCP'] or config['DELAY_SIM'])+\
                        '_Fp='+str(sim_data.plan_freq/1000)+'_Fc='+str(sim_data.ctrl_freq/1000)+'_Fs'+str(sim_data.simu_freq/1000)

# Extract plot data from sim data
plot_data = sim_data.extract_data(frame_of_interest=frame_of_interest)
# Plot results
sim_data.plot_mpc_results(plot_data, which_plots=sim_data.WHICH_PLOTS,
                                    PLOT_PREDICTIONS=True, 
                                    pred_plot_sampling=int(sim_data.plan_freq/10),
                                    SAVE=False,
                                    SAVE_DIR=save_dir,
                                    SAVE_NAME=save_name,
                                    AUTOSCALE=False)
# Save optionally
if(config['SAVE_DATA']):
  sim_data.save_data(sim_data, save_name=save_name, save_dir=save_dir)