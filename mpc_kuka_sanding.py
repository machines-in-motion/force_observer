'''
Example script : MPC simulation with KUKA arm 
contact force task
'''

import crocoddyl
import numpy as np
import pinocchio as pin
np.set_printoptions(precision=4, linewidth=180)
import pin_utils, mpc_utils

from bullet_utils.env import BulletEnvWithGround
from robot_properties_kuka.iiwaWrapper import IiwaRobot
import pybullet as p
import pinocchio as pin
from ContactModel6D import DAMRigidContact6D

from estimator import Estimator

# OCP parameters
T            = 12
dt           = 1e-2
force_weight = 1.

def solveOCP(q, v, ddp, nb_iter, node_id_reach, target_reach, node_id_contact, node_id_track, node_id_circle, force_weight, TASK_PHASE, target_force):
        x = np.concatenate([q, v])
        ddp.problem.x0 = x
        # Warm-start
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
        if(TASK_PHASE == 2):
            # If node id is valid
            if(node_id_contact <= ddp.problem.T and node_id_contact >= 0):
                # Updates nodes between node_id and terminal node 
                for k in range( node_id_contact, ddp.problem.T+1, 1 ):  
                    m[k].differential.costs.costs["translation"].active = True
                    m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                    m[k].differential.costs.costs["translation"].weight = 2.
                    # activate contact and force cost
                    m[k].differential.contacts.changeContactStatus("contact", True)
                    if(k < ddp.problem.T):
                        fref = pin.Force(np.array([0., 0., target_force[k], 0., 0., 0.]))
                        m[k].differential.costs.costs["force"].active = True
                        m[k].differential.costs.costs["force"].cost.residual.reference = fref
        # Update OCP for circle phase
        if(TASK_PHASE == 3):
            # If node id is valid
            if(node_id_circle <= ddp.problem.T and node_id_circle >= 0):
                # Updates nodes between node_id and terminal node
                for k in range( node_id_circle, ddp.problem.T+1, 1 ):
                    m[k].differential.costs.costs["translation"].active = True
                    m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                    m[k].differential.costs.costs["translation"].cost.activation.weights = np.array([1., 1., 0.])
                    m[k].differential.costs.costs["translation"].weight = 10.
                    # activate contact and force cost
                    m[k].differential.contacts.changeContactStatus("contact", True)
                    if(k < ddp.problem.T):
                        fref = pin.Force(np.array([0., 0., target_force[k], 0., 0., 0.]))
                        m[k].differential.costs.costs["force"].active = True
                        m[k].differential.costs.costs["force"].cost.residual.reference = fref
        # get predicted force from rigid model (careful : expressed in LOCAL !!!)
        # j_wrenchpred = ddp.problem.runningDatas[0].differential.multibody.contacts.contacts['contact'].f
        # fpred = jMc.actInv(j_wrenchpred).linear
        # Solve OCP 
        ddp.solve(xs_init, us_init, maxiter=nb_iter, isFeasible=False)
        ddp_iter = ddp.iter
        return ddp.us, ddp.xs, ddp.K




# # # # # # # # # # # # # # # # # # #
### LOAD ROBOT MODEL and SIMU ENV ### 
# # # # # # # # # # # # # # # # # # # 
# Simulation environment
env = BulletEnvWithGround(p.GUI, dt=1e-3)
# env = BulletEnvWithGround(p.DIRECT, dt=1e-3)
# Robot simulator 
robot_simulator = IiwaRobot()
# Extract robot model
nq = robot_simulator.pin_robot.model.nq
nv = robot_simulator.pin_robot.model.nv
nu = nq; nx = nq+nv
q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
v0 = np.zeros(nv)
x0 = np.concatenate([q0, v0])
# Add robot to simulation and initialize
env.add_robot(robot_simulator)
robot_simulator.reset_state(q0, v0)
robot_simulator.forward_robot(q0, v0)
print("[PyBullet] Created robot (id = "+str(robot_simulator.robotId)+")")

# Display contact surface 
contact_frame_id = robot_simulator.pin_robot.model.getFrameId("contact")
contact_frame_placement = robot_simulator.pin_robot.data.oMf[contact_frame_id]
offset = 0.03348 
contact_frame_placement.translation = contact_frame_placement.act(np.array([0., 0., offset]))
mpc_utils.display_contact_surface(contact_frame_placement, with_collision=True, TILT=[0., 0., 0.])



# # # # # # # # # # # # # # #
###  SETUP CROCODDYL OCP  ###
# # # # # # # # # # # # # # #

# State and actuation model
state = crocoddyl.StateMultibody(robot_simulator.pin_robot.model)
actuation = crocoddyl.ActuationModelFull(state)

# Running and terminal cost models
runningModels = []
for i in range(T):  
    # Create DAM 
    delta_f = np.zeros(6)
    delta_f[2] = -0
    dam = DAMRigidContact6D(state, 
                            actuation, 
                            crocoddyl.ContactModelMultiple(state, actuation.nu), 
                            crocoddyl.CostModelSum(state, nu=actuation.nu), 
                            contact_frame_id,
                            delta_f)
    # Create IAM 
    runningModels.append(crocoddyl.IntegratedActionModelEuler(dam, stepTime=dt))

    # Contact model 
    contact_position = robot_simulator.pin_robot.data.oMf[contact_frame_id].copy()
    baumgarte_gains  = np.array([0., 50.])
    runningModels[i].differential.contacts.addContact('contact', crocoddyl.ContactModel6D(state, contact_frame_id, contact_position, baumgarte_gains) , active=True)

    # Cost model
        # Control regularization cost
    uResidual = crocoddyl.ResidualModelContactControlGrav(state)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    runningModels[i].differential.costs.addCost("ctrlRegGrav", uRegCost, 1e-2)
        # State regularization cost
    xResidual = crocoddyl.ResidualModelState(state, x0)
    xRegCost = crocoddyl.CostModelResidual(state, xResidual)
    runningModels[i].differential.costs.addCost("stateReg", xRegCost, 1e-2)
        # End-effector frame force cost
    desired_wrench = np.array([0., 0., -100., 0., 0., 0.])
    frameForceResidual = crocoddyl.ResidualModelContactForce(state, contact_frame_id, pin.Force(desired_wrench), 6, actuation.nu)
    contactForceCost = crocoddyl.CostModelResidual(state, frameForceResidual)
    runningModels[i].differential.costs.addCost("force", contactForceCost, force_weight)
        # End-effector translation cost
    desired_translation = robot_simulator.pin_robot.data.oMf[contact_frame_id].translation.copy()
    frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, contact_frame_id, desired_translation, actuation.nu)
    frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)
    runningModels[i].differential.costs.addCost("translation", frameTranslationCost, 1.)
    
# Terminal DAM
dam_t = DAMRigidContact6D(state, 
                          actuation, 
                          crocoddyl.ContactModelMultiple(state, actuation.nu), 
                          crocoddyl.CostModelSum(state, nu=actuation.nu),
                          contact_frame_id, 
                          delta_f)
terminalModel = crocoddyl.IntegratedActionModelEuler(dam_t, stepTime=0.)
# Terminal contact model 
contact_position = robot_simulator.pin_robot.data.oMf[contact_frame_id].copy()
baumgarte_gains  = np.array([0., 50.])
terminalModel.differential.contacts.addContact('contact', crocoddyl.ContactModel6D(state, contact_frame_id, contact_position, baumgarte_gains) , active=True)
# Terminal cost model 
    # State reg
xResidualTerminal = crocoddyl.ResidualModelState(state, x0)
xRegCostTerminal  = crocoddyl.CostModelResidual(state, xResidual)
terminalModel.differential.costs.addCost("stateReg", xRegCost, 1e-2)
    # Frame translation
frameTranslationResidualTerminal = crocoddyl.ResidualModelFrameTranslation(state, contact_frame_id, desired_translation, actuation.nu)
frameTranslationCostTerminal     = crocoddyl.CostModelResidual(state, frameTranslationResidual)
terminalModel.differential.costs.addCost("translation", frameTranslationCost, 1e-2)

# Create shooting problem & solver
problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
ddp = crocoddyl.SolverFDDP(problem)
# ddp.setCallbacks([crocoddyl.CallbackLogger(),
#                 crocoddyl.CallbackVerbose()])

# Warm start + solve
xs_init = [x0 for i in range(T+1)]
us_init = ddp.problem.quasiStatic(xs_init[:-1])

#   # !!! Deactivate all costs & contact models initially !!!
#   for k,m in enumerate(models):
#       # logger.debug(str(m.differential.costs.active.tolist()))
#       m.differential.costs.costs["translation"].active = False
#       if(k < T):
#            m.differential.costs.costs["force"].active = False
#            m.differential.costs.costs["force"].cost.residual.reference = pin.Force.Zero()
#       m.differential.contacts.changeContactStatus("contact", False)

ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)


# Setup tracking problem with circle ref EE trajectory + Warm start state = IK of circle trajectory
RADIUS = 0.7
OMEGA  = 3.
xs_init = [] 
us_init = []
# Force trajectory
F_MIN = 5.
F_MAX = 50
N_total = 10000 
N_min  = 5
N_ramp = N_min + 10
target_force_traj = np.zeros( (N_total, 3) )
target_force_traj[0:N_min*T, 2] = F_MIN
target_force_traj[N_min*T:N_ramp*T, 2] = [F_MIN + (F_MAX - F_MIN)*i/((N_ramp-N_min)*T) for i in range((N_ramp-N_min)*T)]
target_force_traj[N_ramp*T:, 2] = F_MAX
target_force = np.zeros(T+1)
force_weight = np.asarray(1.)
# Circle trajectory 
N_total_pos = 10000 #int((config['T_tot'] - config['T_REACH'])/dt + T)
N_circle = 10000 #int((config['T_tot'] - config['T_CIRCLE'])/dt) + T
target_position_traj = np.zeros( (N_total_pos, 3) )
target_velocity_traj = np.zeros( (N_total_pos, 3) )
# absolute desired position
pdes = np.array([0.6, 0., 0.1]) 
target_position_traj[0:N_circle, :] = [np.array([pdes[0] + RADIUS * np.sin(i*dt*OMEGA), 
                                                    pdes[1] + RADIUS * (1-np.cos(i*dt*OMEGA)),
                                                    pdes[2]]) for i in range(N_circle)]
target_velocity_traj[0:N_circle, :] = [np.array([RADIUS * OMEGA * np.cos(i*dt*OMEGA), 
                                                    RADIUS * OMEGA * np.sin(i*dt*OMEGA),
                                                    0.]) for i in range(N_circle)]
target_position_traj[N_circle:, :] = target_position_traj[N_circle-1,:]
target_velocity_traj[N_circle:, :] = np.zeros(3)
target_position = np.zeros((T+1, 3)) 
target_position[:,:] = pdes.copy()
target_velocity = np.zeros((T+1, 3)) 
q_ws = q0



# # # # # # # # # # # #
###  MPC SIMULATION ###
# # # # # # # # # # # #
# OCP parameters
ocp_params = {}
ocp_params['N_h']          = T
ocp_params['dt']           = dt
ocp_params['maxiter']      = 10
ocp_params['pin_model']    = robot_simulator.pin_robot.model
ocp_params['armature']     = np.zeros(nv)
ocp_params['contact_frame_id'] = contact_frame_id
ocp_params['id_endeff'] = contact_frame_id
ocp_params['active_costs'] = ddp.problem.runningModels[0].differential.costs.active.tolist()


# Simu parameters
sim_params = {}
sim_params['sim_freq']  = int(1./env.dt)
sim_params['mpc_freq']  = 1000
sim_params['dt_plan']   = 1./sim_params['mpc_freq']
sim_params['dt_simu']   = 1./sim_params['sim_freq']
sim_params['T_sim']     = 1.
log_rate = 100
# Horizon in simu cycles
node_id_reach   = -1
node_id_contact = -1
node_id_track   = -1
node_id_circle  = -1
TASK_PHASE      = 0
NH_SIMU   = int(ocp_params['N_h']*ocp_params['dt']*sim_params['sim_freq'])
T_REACH   = int(1.*sim_params['sim_freq'])
T_CONTACT = int(3.*sim_params['sim_freq'])
T_CIRCLE  = int(5.*sim_params['sim_freq'])
OCP_TO_MPC_CYCLES  = 1./(sim_params['dt_plan'] / sim_params['mpc_freq']*ocp_params['dt'])
OCP_TO_SIMU_CYCLES = 1./(sim_params['dt_simu'] / ocp_params['dt'])
print("Size of MPC horizon in simu cycles     = "+str(NH_SIMU))
print("Start of reaching phase in simu cycles = "+str(T_REACH))
print("Start of contact phase in simu cycles  = "+str(T_CONTACT))
print("Start of circle phase in simu cycles   = "+str(T_CIRCLE))
print("OCP to PLAN time ratio = "+str(OCP_TO_MPC_CYCLES))

# Initialize simulation data 
sim_data = mpc_utils.init_sim_data(sim_params, ocp_params, x0)
sim_data['contact_frame_id'] = contact_frame_id

# Estimation

q_mea_SIM_RATE = q0
v_mea_SIM_RATE = v0
f_mea_SIM_RATE = mpc_utils.get_contact_wrench(robot_simulator, sim_data['contact_frame_id'])
df_prior = np.array([25, 20, -20, 0, -5, 1])
df_prior = np.zeros(6)
df_list = []
F_list = []
F_GT_list = []
F_mes_list = []
F = np.zeros(6)

force_estimator = Estimator(baumgarte_gains)


# Simulate
mpc_cycle = 0
for i in range(sim_data['N_sim']): 

    if(i%log_rate==0): 
        print("\n SIMU step "+str(i)+"/"+str(sim_data['N_sim'])+"\n")

    # Solve OCP if we are in a planning cycle (MPC/planning frequency)
    if(i%int(sim_params['sim_freq']/sim_params['mpc_freq']) == 0):
        # Set x0 to measured state 
        ddp.problem.x0 = sim_data['state_mea_SIM_RATE'][i, :]
        # Warm start using previous solution
        xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]
        xs_init[0] = sim_data['state_mea_SIM_RATE'][i, :]
        us_init = list(ddp.us[1:]) + [ddp.us[-1]] 


        # # # # # # # # # 
        # # Update OCP  #
        # # # # # # # # # 
        time_to_reach   = int(i - T_REACH)
        time_to_contact = int(i - T_CONTACT)
        time_to_circle  = int(i - T_CIRCLE)

        if(time_to_reach == 0): 
            print("Entering reaching phase")
        # If tracking phase enters the MPC horizon, start updating models from the end with tracking models      
        if(0 <= time_to_reach and time_to_reach <= NH_SIMU):
            TASK_PHASE = 1
            # If current time matches an OCP node 
            if(time_to_reach%OCP_TO_SIMU_CYCLES == 0):
                # Select IAM
                node_id_reach = T - int(time_to_reach/OCP_TO_SIMU_CYCLES)

        if(time_to_contact == 0): 
            # Record end-effector position at the time of the contact switch
            position_at_contact_switch = robot_simulator.pin_robot.data.oMf[contact_frame_id].translation.copy()
            target_position[:,:] = position_at_contact_switch.copy()
            print("Entering contact phase")
        # If contact phase enters horizon start updating models from the the end with contact models
        if(0 <= time_to_contact and time_to_contact <= NH_SIMU):
            TASK_PHASE = 2
            # If current time matches an OCP node 
            if(time_to_contact%OCP_TO_SIMU_CYCLES == 0):
                # Select IAM
                node_id_contact = T - int(time_to_contact/OCP_TO_SIMU_CYCLES)

        if(0 <= time_to_contact and time_to_contact%OCP_TO_SIMU_CYCLES == 0):
            ti  = int(time_to_contact/OCP_TO_SIMU_CYCLES)
            tf  = ti + T+1
            target_force = target_force_traj[ti:tf,2]

        if(time_to_circle == 0): 
            print("Entering circle phase")
        # If circle tracking phase enters the MPC horizon, start updating models from the end with tracking models      
        if(0 <= time_to_circle and time_to_circle <= NH_SIMU):
            TASK_PHASE = 3
            # If current time matches an OCP node 
            if(time_to_circle%OCP_TO_SIMU_CYCLES == 0):
                # Select IAM
                node_id_circle = T - int(time_to_circle/OCP_TO_SIMU_CYCLES)

        if(0 <= time_to_circle and time_to_circle%OCP_TO_SIMU_CYCLES == 0):
            # set position refs over current horizon
            ti  = int(time_to_circle/OCP_TO_SIMU_CYCLES)
            tf  = ti + T+1
            # Target in (x,y)  = circle trajectory + offset to start from current position instead of absolute target
            offset_xy = position_at_contact_switch[:2] - pdes[:2]
            target_position[:,:2] = target_position_traj[ti:tf,:2] + offset_xy
            # Target in z is fixed to the anchor at switch (equals absolute target if RESET_ANCHOR = False)
            # No position tracking in z : redundant with zero activation weight on z
            target_position[:,2]  = robot_simulator.pin_robot.data.oMf[contact_frame_id].translation[2].copy()
            # Record target signals                
            target_velocity[:,:2] = target_velocity_traj[ti:tf,:2] 
            target_velocity[:,2]  = 0.

        # ########################## UPDATE MODEL
        # running_DAM = DAMRigidContact6D(state, actuation, contactModel, runningCostModel, contact_frame_id, df_prior)
        # # terminal_DAM = DAMRigidContact6D(state, actuation, contactModel, terminalCostModel, contact_frame_id, delta_f)
        # terminal_DAM = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, terminalCostModel, inv_damping=0., enable_force=True)

        # dt = 1e-2
        # runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
        # terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)
        # # # Optionally add armature to take into account actuator's 
        # # Create the shooting problem
        # T = 12
        # problem = crocoddyl.ShootingProblem(ddp.problem.x0, [runningModel] * T, terminalModel)

        # ddp = crocoddyl.SolverFDDP(problem)





        # Solve OCP & record MPC predictions
        q = sim_data['state_mea_SIM_RATE'][i, :nq]
        v = sim_data['state_mea_SIM_RATE'][i, nq:]
        solveOCP(q, v, ddp, ocp_params['maxiter'], node_id_reach, target_position, node_id_contact, node_id_track, node_id_circle, force_weight, TASK_PHASE, target_force)
        ddp.solve(xs_init, us_init, maxiter=ocp_params['maxiter'], isFeasible=False)
        sim_data['state_pred'][mpc_cycle, :, :]  = np.array(ddp.xs)
        sim_data['ctrl_pred'][mpc_cycle, :, :]   = np.array(ddp.us)
        sim_data ['force_pred'][mpc_cycle, :, :] = np.array([ddp.problem.runningDatas[i].differential.multibody.contacts.contacts['contact'].f.vector for i in range(ocp_params['N_h'])])
        # Extract relevant predictions for interpolations
        x_curr = sim_data['state_pred'][mpc_cycle, 0, :]    # x0* = measured state    (q^,  v^ )
        x_pred = sim_data['state_pred'][mpc_cycle, 1, :]    # x1* = predicted state   (q1*, v1*) 
        u_curr = sim_data['ctrl_pred'][mpc_cycle, 0, :]     # u0* = optimal control   (tau0*)
        f_curr = sim_data['force_pred'][mpc_cycle, 0, :]
        f_pred = sim_data['force_pred'][mpc_cycle, 1, :]
        # Record costs references
        q = sim_data['state_pred'][mpc_cycle, 0, :sim_data['nq']]
        sim_data['ctrl_ref'][mpc_cycle, :]       = pin_utils.get_u_grav(q, ddp.problem.runningModels[0].differential.pinocchio, ocp_params['armature'])
        sim_data['f_ee_ref'][mpc_cycle, :]       = ddp.problem.runningModels[0].differential.costs.costs['force'].cost.residual.reference.vector
        sim_data['state_ref'][mpc_cycle, :]      = ddp.problem.runningModels[0].differential.costs.costs['stateReg'].cost.residual.reference


        # Select reference control and state for the current MPC cycle
        x_ref_MPC_RATE  = x_curr + sim_data['ocp_to_mpc_ratio'] * (x_pred - x_curr)
        u_ref_MPC_RATE  = u_curr 
        f_ref_MPC_RATE  = f_curr + sim_data['ocp_to_mpc_ratio'] * (f_pred - f_curr)
        if(mpc_cycle==0):
            sim_data['state_des_MPC_RATE'][mpc_cycle, :]   = x_curr  
        sim_data['ctrl_des_MPC_RATE'][mpc_cycle, :]    = u_ref_MPC_RATE   
        sim_data['state_des_MPC_RATE'][mpc_cycle+1, :] = x_ref_MPC_RATE    
        sim_data['force_des_MPC_RATE'][mpc_cycle, :] = f_ref_MPC_RATE    
        
        # Increment planning counter
        mpc_cycle += 1
        

        # Select reference control and state for the current SIMU cycle
        x_ref_SIM_RATE  = x_curr + sim_data['ocp_to_mpc_ratio'] * (x_pred - x_curr)
        u_ref_SIM_RATE  = u_curr 
        f_ref_SIM_RATE  = f_curr + sim_data['ocp_to_mpc_ratio'] * (f_pred - f_curr)

        # First prediction = measurement = initialization of MPC
        if(i==0):
            sim_data['state_des_SIM_RATE'][i, :]   = x_curr  
        sim_data['ctrl_des_SIM_RATE'][i, :]    = u_ref_SIM_RATE  
        sim_data['state_des_SIM_RATE'][i+1, :] = x_ref_SIM_RATE 
        sim_data['force_des_SIM_RATE'][i, :] = f_ref_SIM_RATE 

        #  Send output of actuation torque to the RBD simulator 
        robot_simulator.send_joint_command(u_ref_SIM_RATE * 0.8 + 0.01*robot_simulator.pin_robot.model.effortLimit)
        env.step()



        # Estimation 1
        q_old = q_mea_SIM_RATE.copy()
        v_old = v_mea_SIM_RATE.copy()
        

        theta = 1.
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[1, 0, 0], [0, c, -s], [0., s, c]])  
        # print(R)
        if f_mea_SIM_RATE is not None:
          F_mes = f_mea_SIM_RATE.copy()
          # F_mes[:3] = R @ F_mes[:3]
          # F_mes[2] += 50 + 20*np.sin(i*0.1)
          # F_mes += np.random.multivariate_normal(np.zeros(6), 4*np.eye(6))

        # Measure new state from simulation 
        q_mea_SIM_RATE, v_mea_SIM_RATE = robot_simulator.get_state()
        # v_mea_SIM_RATE += np.random.multivariate_normal(np.zeros(7), 0.005*np.eye(7))


        # Estimation 2
        if f_mea_SIM_RATE is not None:
          a = (v_mea_SIM_RATE - v_old) / env.dt
          F, df_prior = force_estimator.estimate(robot_simulator.pin_robot, q_old, v_old, a, u_ref_SIM_RATE, df_prior, F_mes)
        df_list.append(df_prior)
        F_list.append(F)
        # print(df_prior)
        if f_mea_SIM_RATE is None:
          F_GT_list.append(np.zeros(6))
          F_mes_list.append(np.zeros(6))
        else:
          F_GT_list.append(f_mea_SIM_RATE)
          F_mes_list.append(F_mes)
        # print(df_prior)
        # df_prior = np.zeros(6)
        # Update pinocchio model
        robot_simulator.forward_robot(q_mea_SIM_RATE, v_mea_SIM_RATE)
        f_mea_SIM_RATE = mpc_utils.get_contact_wrench(robot_simulator, sim_data['contact_frame_id']) #+ np.random.multivariate_normal(np.zeros(6), 4*np.eye(6))
        # Record data (unnoised)
        x_mea_SIM_RATE = np.concatenate([q_mea_SIM_RATE, v_mea_SIM_RATE]).T 
        sim_data['state_mea_SIM_RATE'][i+1, :] = x_mea_SIM_RATE
        sim_data['force_mea_SIM_RATE'][i, :] = f_mea_SIM_RATE



import matplotlib.pyplot as plt
df_list = np.array(df_list)
F_list = np.array(F_list)
F_GT_list = np.array(F_GT_list)
F_mes_list = np.array(F_mes_list)
fig, ax = plt.subplots(3, 2, figsize=(19.2,10.8), sharex='col') 
# Plot endeff
xyz = ['x', 'y', 'z']

time_lin = np.linspace(0, 1e-3*len(df_list), len(df_list))
for i in range(3):
  ax[i, 0].plot(time_lin, df_list[:,i], label="df")
  ax[i, 0].grid()
  ax[i, 0].set_ylabel('$\\Delta F^{EE}_%s$  (N)'%xyz[i])

  ax[i, 1].plot(time_lin, df_list[:,i+3], label="df")
  ax[i, 1].grid()
  ax[i, 1].set_ylabel('$\\Delta \\tau^{EE}_%s$  (N)'%xyz[i])

ax[2, 0].set_xlabel("Time [s]")
ax[2, 1].set_xlabel("Time [s]")
# ax[0, 0].legend()


plot_data = mpc_utils.extract_plot_data_from_sim_data(sim_data)

mpc_utils.plot_mpc_results(plot_data, which_plots=['all'], PLOT_PREDICTIONS=True, pred_plot_sampling=int(sim_params['mpc_freq']/10), AUTOSCALE=True)

