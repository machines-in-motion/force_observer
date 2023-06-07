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
from ContactModel import DAMRigidContact
import sobec
from estimator import Estimator

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
mpc_utils.display_contact_surface(contact_frame_placement, bullet_endeff_ids=robot_simulator.bullet_endeff_ids) #, with_collision=True, TILT=[0., 0., 0.])

# # # # # # # # # # # # # # #
###  SETUP CROCODDYL OCP  ###
# # # # # # # # # # # # # # #
# State and actuation model
state = crocoddyl.StateMultibody(robot_simulator.pin_robot.model)
actuation = crocoddyl.ActuationModelFull(state)
# Running and terminal cost models
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)
# Contact model 
contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)
# Create 3D contact on the en-effector frame
nc = 6
baumgarte_gains  = np.array([0., 50.])
if(nc == 3):
    contactItem = sobec.ContactModel3D(state, contact_frame_id, robot_simulator.pin_robot.data.oMf[contact_frame_id].translation, baumgarte_gains, pin.LOCAL) 
elif(nc == 6):
    contactItem = sobec.ContactModel6D(state, contact_frame_id, robot_simulator.pin_robot.data.oMf[contact_frame_id], baumgarte_gains, pin.LOCAL) 
elif(nc == 1):
    contactItem = sobec.ContactModel1D(state, contact_frame_id, robot_simulator.pin_robot.data.oMf[contact_frame_id].translation, actuation.nu, baumgarte_gains, sobec.Vector3MaskType.z, pin.LOCAL) 
else: pass
# Populate contact model with contacts
contactModel.addContact("contact", contactItem, active=True)
# Create cost terms 
  # Control regularization cost
uResidual = crocoddyl.ResidualModelContactControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
  # State regularization cost
xResidual = crocoddyl.ResidualModelState(state, x0)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
  # End-effector frame force cost
desired_wrench = np.array([0., 0., -100., 0., 0., 0.])
frameForceResidual = crocoddyl.ResidualModelContactForce(state, contact_frame_id, pin.Force(desired_wrench), nc, actuation.nu)
contactForceCost = crocoddyl.CostModelResidual(state, frameForceResidual)
# Populate cost models with cost terms
runningCostModel.addCost("stateReg", xRegCost, 1e-2)
runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
runningCostModel.addCost("force", contactForceCost, 10.)
terminalCostModel.addCost("stateReg", xRegCost, 1e-2)
# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
# running_DAM = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, runningCostModel, inv_damping=0., enable_force=True)
# terminal_DAM = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, terminalCostModel, inv_damping=0., enable_force=True)
if(nc == 1):
    delta_f = np.zeros(3) 
else:
    delta_f = np.zeros(nc) 
running_DAM = DAMRigidContact(state, actuation, contactModel, runningCostModel, contact_frame_id, delta_f)
# terminal_DAM = DAMRigidContact(state, actuation, contactModel, terminalCostModel, contact_frame_id, delta_f)
terminal_DAM = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, terminalCostModel, inv_damping=0., enable_force=True)



# Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
dt = 1e-2
runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)
# # Optionally add armature to take into account actuator's inertia
runningModel.differential.armature = np.array([0, 0., 0., 0., 0, 0., 0.])
terminalModel.differential.armature = np.array([0., 0., 0., 0., 0., 0., 0.])
# Create the shooting problem
T = 12
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
# Create solver + callbacks
ddp = crocoddyl.SolverFDDP(problem)
# ddp.setCallbacks([crocoddyl.CallbackLogger(),
#                 crocoddyl.CallbackVerbose()])
# Warm start : initial state + gravity compensation
xs_init = [x0 for i in range(T+1)]
us_init = ddp.problem.quasiStatic(xs_init[:-1])
# Solve
ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)




# # # # # # # # # # # #
###  MPC SIMULATION ###
# # # # # # # # # # # #
# OCP parameters
ocp_params = {}
ocp_params['N_h']          = T
ocp_params['dt']           = dt
ocp_params['maxiter']      = 10
ocp_params['pin_model']    = robot_simulator.pin_robot.model
ocp_params['armature']     =  runningModel.differential.armature
ocp_params['id_endeff']    = contact_frame_id
ocp_params['active_costs'] = ddp.problem.runningModels[0].differential.costs.active.tolist()

# Simu parameters
sim_params = {}
sim_params['sim_freq']  = int(1./env.dt)
sim_params['mpc_freq']  = 1000
sim_params['T_sim']     = 1.
log_rate = 100

# Initialize simulation data 
sim_data = mpc_utils.init_sim_data(sim_params, ocp_params, x0)



# Estimation

q_mea_SIM_RATE = q0
v_mea_SIM_RATE = v0
f_mea_SIM_RATE = mpc_utils.get_contact_wrench(robot_simulator, sim_data['id_endeff'])
# df_prior = np.array([25, 20, -20, 0, -5, 1])
df_prior = np.zeros(6)
df_list = []
F_list = []
F_GT_list = []
F_mes_list = []
F = np.zeros(6)

force_estimator = Estimator(robot_simulator.pin_robot, nc, contact_frame_id, baumgarte_gains)


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



        ########################## UPDATE MODEL
        # running_DAM = DAMRigidContact(state, actuation, contactModel, runningCostModel, contact_frame_id, df_prior)
        # # terminal_DAM = DAMRigidContact(state, actuation, contactModel, terminalCostModel, contact_frame_id, delta_f)
        # terminal_DAM = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, terminalCostModel, inv_damping=0., enable_force=True)

        # dt = 1e-2
        # runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
        # terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)
        # # # Optionally add armature to take into account actuator's 
        # # Create the shooting problem
        # T = 12
        # problem = crocoddyl.ShootingProblem(ddp.problem.x0, [runningModel] * T, terminalModel)

        # ddp = crocoddyl.SolverFDDP(problem)
        for m in ddp.problem.runningModels:
           m.differential.delta_f = df_prior[:nc]*0.


        # Solve OCP & record MPC predictions
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
        robot_simulator.send_joint_command(u_ref_SIM_RATE) # * 0.8 + 0.01*robot_simulator.pin_robot.model.effortLimit)
        env.step()



        # Estimation 1
        q_old = q_mea_SIM_RATE.copy()
        v_old = v_mea_SIM_RATE.copy()
        

        theta = 1.
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[1, 0, 0], [0, c, -s], [0., s, c]])  
        # print(R)
        # if f_mea_SIM_RATE is not None:
        #   F_mes = f_mea_SIM_RATE[:nc].copy()
          # F_mes[:3] = R @ F_mes[:3]
          # F_mes[2] += 50 + 20*np.sin(i*0.1)
          # F_mes += np.random.multivariate_normal(np.zeros(6), 4*np.eye(6))

        # Measure new state from simulation 
        q_mea_SIM_RATE, v_mea_SIM_RATE = robot_simulator.get_state()
        # v_mea_SIM_RATE += np.random.multivariate_normal(np.zeros(7), 0.005*np.eye(7))


        # Estimation 2
        if f_mea_SIM_RATE is not None:
          a = (v_mea_SIM_RATE - v_old) / env.dt
          F, df_prior = force_estimator.estimate(q_old, v_old, a, u_ref_SIM_RATE, df_prior[:nc], f_mea_SIM_RATE[:nc])
          if(nc == 3):
            df_prior = np.concatenate([df_prior, np.zeros(3)])
        df_list.append(df_prior)
        F_list.append(F)
        # print(df_prior)
        if f_mea_SIM_RATE is None:
          F_GT_list.append(np.zeros(6))
          F_mes_list.append(np.zeros(6))
        else:
          F_GT_list.append(f_mea_SIM_RATE)
          F_mes_list.append(f_mea_SIM_RATE)
        # print(df_prior)
        # df_prior = np.zeros(6)
        # Update pinocchio model
        robot_simulator.forward_robot(q_mea_SIM_RATE, v_mea_SIM_RATE)
        
        # !!! PyBullet forces come in LWA by default : transform to LOCAL if necessary
        f_mea_SIM_RATE_world = robot_simulator.end_effector_forces(pin.LOCAL)[1][0]
        lwaMc = robot_simulator.pin_robot.data.oMf[sim_data['id_endeff']]
        f_mea_SIM_RATE = lwaMc.actInv(pin.Force(f_mea_SIM_RATE_world)).vector


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

mpc_utils.plot_mpc_results(plot_data, which_plots=['f'], PLOT_PREDICTIONS=True, pred_plot_sampling=int(sim_params['mpc_freq']/10), AUTOSCALE=True)

