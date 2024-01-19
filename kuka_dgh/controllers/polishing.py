import numpy as np
import pinocchio as pin 
import mim_solvers
import time

import sys
sys.path.append("../demos/utils")
from ocp_utils import OptimalControlProblemClassicalWithObserver
from force_observer import ForceEstimator, MHForceEstimator, TorqueEstimator

import croco_mpc_utils.pinocchio_utils as pin_utils
from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

from utils.reduced_model import get_controlled_joint_ids



def solveOCP(q, v, solver, nb_iter, target_reach, force_weight, TASK_PHASE, target_force):
        t = time.time()
        # Update initial state + warm-start
        
        x = np.concatenate([q, v])
        solver.problem.x0 = x
        

        xs_init = list(solver.xs[1:]) + [solver.xs[-1]]
        xs_init[0] = x
        us_init = list(solver.us[1:]) + [solver.us[-1]] 
      
            
        # Update OCP for reaching phase
        if(TASK_PHASE == 1):
            for k in range(solver.problem.T):
                solver.problem.runningModels[k].differential.costs.costs["translation"].active = True
                solver.problem.runningModels[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
            solver.problem.terminalModel.differential.costs.costs["translation"].active = True
            solver.problem.terminalModel.differential.costs.costs["translation"].cost.residual.reference = target_reach[-1]    
            
        # Update OCP for "increase weights" phase
        if(TASK_PHASE == 2):
            for k in range(solver.problem.T):
                w = min(2.*(k + 1.) , 5)
                solver.problem.runningModels[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                solver.problem.runningModels[k].differential.costs.costs["translation"].weight = w
            solver.problem.terminalModel.differential.costs.costs["translation"].cost.residual.reference = target_reach[-1]
            solver.problem.terminalModel.differential.costs.costs["translation"].weight = w
        # Update OCP for contact phase
        if(TASK_PHASE == 3):
            for k in range(solver.problem.T):  
                solver.problem.runningModels[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                solver.problem.runningModels[k].differential.costs.costs["translation"].cost.activation.weights = np.array([1., 1., 0.])
                solver.problem.runningModels[k].differential.costs.costs["translation"].weight = 60. 
                solver.problem.runningModels[k].differential.costs.costs['rotation'].active = True
                solver.problem.runningModels[k].differential.costs.costs['rotation'].cost.residual.reference = pin.utils.rpyToMatrix(np.pi, 0, np.pi)
                solver.problem.runningModels[k].differential.contacts.changeContactStatus("contact", True)
                solver.problem.runningModels[k].differential.costs.costs["force"].weight = force_weight
                solver.problem.runningModels[k].differential.costs.costs["force"].cost.residual.reference = pin.Force(np.array([0., 0., target_force[k], 0., 0., 0.]))         
            solver.problem.terminalModel.differential.costs.costs["translation"].cost.residual.reference = target_reach[-1]
            solver.problem.terminalModel.differential.costs.costs["translation"].cost.activation.weights = np.array([1., 1., 0.])
            solver.problem.terminalModel.differential.costs.costs["translation"].weight = 60. 
            solver.problem.terminalModel.differential.costs.costs['rotation'].active = True
            solver.problem.terminalModel.differential.costs.costs['rotation'].cost.residual.reference = pin.utils.rpyToMatrix(np.pi, 0, np.pi)
            solver.problem.terminalModel.differential.contacts.changeContactStatus("contact", True)    
                    
        # Update OCP for circle phase
        if(TASK_PHASE == 4):
            for k in range(solver.problem.T):
                solver.problem.runningModels[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                # update force ref
                solver.problem.runningModels[k].differential.costs.costs["force"].cost.residual.reference = pin.Force(np.array([0., 0., target_force[k], 0., 0., 0.]))
            solver.problem.terminalModel.differential.costs.costs["translation"].cost.residual.reference = target_reach[-1]    
        
        # get predicted force from rigid model (careful : expressed in LOCAL !!!)
        # Solve OCP 
        solver.solve(xs_init, us_init, maxiter=nb_iter, isFeasible=False)
        solve_time = time.time()
        # Send solution to parent process + riccati gains
        return solver.us[0], solver.xs[1], solver.K[0], solve_time - t, solver.iter, solver.KKT



class KukaPolishingMPC:

    def __init__(self, head, pin_robot, config, run_sim, cMs, locked_joints=['A7']):
        """
        Input:
            head              : thread head
            pin_robot         : pinocchio wrapper
            config            : MPC config yaml file
            run_sim           : boolean sim or real
            cMs               : placement of the sensor frame w.r.t. contact frame
            locked_joints     : name of joints that are locked in the full robot model
        """
        self.robot   = pin_robot
        self.head    = head
        self.RUN_SIM = run_sim
        self.joint_positions     = head.get_sensor('joint_positions')
        self.joint_velocities    = head.get_sensor("joint_velocities")
        self.joint_accelerations = head.get_sensor("joint_accelerations")
        if not self.RUN_SIM:
            self.joint_torques     = head.get_sensor("joint_torques_total")
            self.joint_ext_torques = head.get_sensor("joint_torques_external")
            self.joint_cmd_torques = head.get_sensor("joint_torques_commanded")
            self.ft_sensor_wrench  = head.get_sensor("ft_sensor_wrench") # in sensor frame !!
        else: 
            self.ft_sensor_wrench = self.head._sensor__force_plate_force[0]


        self.nq = self.robot.model.nq
        self.nv = self.robot.model.nv

        logger.warning("Controlled model dimensions : ")
        logger.warning(" nq = "+str(self.nq))
        logger.warning(" nv = "+str(self.nv))
        
        # Controlled ids (reduced model)
        self.controlled_joint_ids = get_controlled_joint_ids('iiwa_ft_sensor_shell', locked_joints=locked_joints)
        logger.warning("Controlled joint ids = "+str(self.controlled_joint_ids))
        self.fixed_ids = [i for i in range(7) if i not in self.controlled_joint_ids]
        logger.warning("Fixed joint ids = "+str(self.fixed_ids))
        self.gain_P = 100.
        self.gain_D = 30.
        
        # Config
        self.config = config
        if(self.RUN_SIM):
            self.controlled_joint_ids = range(len(self.controlled_joint_ids)) 
            self.q0 = np.asarray(config['q0'])[self.controlled_joint_ids]    
            self.v0 = self.joint_velocities[self.controlled_joint_ids]    
        else:
            self.q0 = self.joint_positions[self.controlled_joint_ids]    
            self.v0 = self.joint_velocities[self.controlled_joint_ids]  
        self.x0 = np.concatenate([self.q0, self.v0]) 
        
        # Calibration stuff
        self.cMs = cMs
        self.sensorFrameId = self.robot.model.getFrameId('ft_sensor')
        self.contactFrameId = self.robot.model.getFrameId(config['contactFrameName']) 
        logger.warning("Sensor frame id = "+str(self.sensorFrameId))
        logger.warning("Contact frame id = "+str(self.contactFrameId))
        pin.framesForwardKinematics(self.robot.model, self.robot.data, self.q0)

        self.oPc_offset       = np.asarray(self.config['oPc_offset'])
        self.oPc              = np.asarray(config['contactPosition']) + self.oPc_offset
        
        self.Nh = int(self.config['N_h'])
        self.dt_ocp  = self.config['dt']
        self.dt_ctrl = 1./self.config['ctrl_freq']
        self.OCP_TO_CTRL_RATIO = int(self.dt_ocp/self.dt_ctrl)
        self.CTRL_TO_PLAN_RATIO = int(self.config['ctrl_freq']/self.config['plan_freq'])
        # Create OCP
        self.fext0 = [pin.Force.Zero() for _ in self.robot.model.joints]
        if(self.config['pinRefFrame'] == 'LOCAL'):
            self.pinRef = pin.LOCAL
        else:
            self.pinRef = pin.LOCAL_WORLD_ALIGNED

        torque_offset = self.config["USE_DELTA_TAU"] and self.config["INTERNAL"]
        problem = OptimalControlProblemClassicalWithObserver(self.robot, self.config).initialize(self.x0, torque_offset=torque_offset)

        # Initialize the solver
        self.solver                        = mim_solvers.SolverSQP(problem)
        self.solver.regMax                 = 1e6
        self.solver.reg_max                = 1e6
        self.solver.termination_tolerance  = self.config['solver_termination_tolerance']
        self.solver.use_filter_line_search = self.config['use_filter_line_search']
        self.solver.filter_size            = self.config['maxiter']
        
        # !!! Deactivate all costs & contact models initially !!!
        models = list(self.solver.problem.runningModels) + [self.solver.problem.terminalModel]
        for k,m in enumerate(models):
            # logger.debug(str(m.differential.costs.active.tolist()))
            m.differential.costs.costs["translation"].active = False
            m.differential.contacts.changeContactStatus("contact", False)
            m.differential.costs.costs['rotation'].active = False
            m.differential.costs.costs['rotation'].cost.residual.reference = pin.utils.rpyToMatrix(np.pi, 0., np.pi)
            
        # Allocate MPC data
        self.K = self.solver.K[0]
        self.x_des = self.solver.xs[0]
        self.tau_ff = self.solver.us[0]
        self.tau = self.tau_ff.copy() ; self.tau_riccati = np.zeros(self.tau.shape)
        self.fpred = np.zeros(3)

        # Initialize torque measurements 
        if(self.RUN_SIM):
            logger.debug("Initial torque measurement signal : simulation --> use u0 = g(q0)")
            self.u0 = pin_utils.get_tau(self.q0, self.v0, np.zeros(self.nq), self.fext0, self.robot.model, np.zeros(self.robot.model.nq))
            self.joint_torques_total    = self.u0
            self.joint_torques_measured = self.u0
        # DANGER ZONE 
        else:
            logger.warning("Initial torque measurement signal : real robot --> use sensor signal 'joint_torques_total' ")
            self.joint_torques_total    = head.get_sensor("joint_torques_total")
            logger.warning("      >>> Correct minus sign in measured torques ! ")
            self.joint_torques_measured = -self.joint_torques_total 

        # Task phases management & cost parameters
        # Initialize force measurement : Force fs is expressed in the SENSOR frame 
        # Target force in the OCP is expressed in the CENTERED CONTACT frame 
        # Transform the measurement as : SENSOR --> CONTACT --> CENTERED CONTACT 
        self.lwaMc = self.robot.data.oMf[self.contactFrameId].copy()
        self.lwaMc.translation = np.zeros(3)
        if(not self.RUN_SIM):
            f6d_sensor  = pin.Force(self.ft_sensor_wrench)
            # self.contact_force_6d_measured_sensor = f6d_sensor.vector.copy()
            f6d_local   = self.cMs.act(f6d_sensor)
            f6d_world   = self.lwaMc.act(self.cMs.act(f6d_sensor))  
        else:
            #### CAREFUL : PyBullet forces are in WORLD by default...
            f6d_local   = self.lwaMc.actInv(pin.Force(self.ft_sensor_wrench))
            f6d_world   = pin.Force(self.ft_sensor_wrench) 
        if(self.config['pinRefFrame'] == 'LOCAL'):
            logger.warning("LOCAL contact force !")
            self.contact_force_3d_measured = f6d_local.vector[:3].copy()
            self.coef_target_force = -1.
        else:
            logger.warning("LOCAL_WORLD_ALIGNED contact force !")
            self.contact_force_3d_measured  = f6d_world.vector[:3].copy()
            self.coef_target_force = +1.  
             
        # Task phases management & cost parameters
        F_MIN = 5.
        F_MAX = self.config['frameForceRef'][2]
        N_total = int((self.config['T_tot'] - self.config['T_CONTACT'])/self.dt_ctrl + self.Nh*self.OCP_TO_CTRL_RATIO)

        N_ramp = int((self.config['T_RAMP'] - self.config['T_CONTACT']) / self.dt_ctrl)
        self.target_force_traj = np.zeros((N_total, 3))
        self.target_force_traj[:N_ramp, 2] = [F_MIN + (F_MAX - F_MIN)*i/N_ramp for i in range(N_ramp)]
        self.target_force_traj[N_ramp:, 2] = F_MAX
        # self.target_force_traj[N_sinus*self.Nh:, 2] = [F_MAX + 20.*np.round(np.sin(0.01 * (2*np.pi/self.Nh) * i/2 )  ) for i in range(N_total-N_sinus*self.Nh)]
        # plt.plot(self.target_force_traj)
        # plt.show()
        self.target_force = np.zeros(self.Nh+1)
        self.force_weight = self.config['frameForceWeight']

        # Circle trajectory 
        N_total_pos = int((self.config['T_tot'] - self.config['T_REACH'])/self.dt_ctrl + self.Nh*self.OCP_TO_CTRL_RATIO)
        N_circle    = int((self.config['T_tot'] - self.config['T_CIRCLE'])/self.dt_ctrl + self.Nh*self.OCP_TO_CTRL_RATIO )
        self.target_position_traj = np.zeros( (N_total_pos, 3) )
        # absolute desired position
        self.oPc_offset = np.asarray(self.config['oPc_offset'])
        self.pdes = np.asarray(self.config['contactPosition']) + self.oPc_offset
        
        PHASE_TIME = 6283 # in cycles
        # SLOW
        radius = 0.07 ; omega = 1.
        self.target_position_traj[0:PHASE_TIME, :] = [np.array([self.pdes[0] + radius * (1-np.cos(i*self.dt_ctrl*omega)), 
                                                                self.pdes[1] - radius * np.sin(i*self.dt_ctrl*omega),
                                                                self.pdes[2]]) for i in range(PHASE_TIME)]
        # MEDIUM
        omega = 3.
        self.target_position_traj[PHASE_TIME:2*PHASE_TIME, :] = [np.array([self.pdes[0] + radius * (1-np.cos(i*self.dt_ctrl*omega)), 
                                                                           self.pdes[1] - radius * np.sin(i*self.dt_ctrl*omega),
                                                                           self.pdes[2]]) for i in range(PHASE_TIME)]
        # FAST
        omega = 6.
        self.target_position_traj[2*PHASE_TIME:3*PHASE_TIME, :] = [np.array([self.pdes[0] + radius * (1-np.cos(i*self.dt_ctrl*omega)), 
                                                                             self.pdes[1] - radius * np.sin(i*self.dt_ctrl*omega),
                                                                             self.pdes[2]]) for i in range(PHASE_TIME)]
        self.target_position_traj[3*PHASE_TIME:, :] = self.target_position_traj[3*PHASE_TIME-1,:]
        # import matplotlib.pyplot as plt
        # plt.plot(self.target_position_traj, label='pos')
        # plt.show()
        # Targets over one horizon (initially = absolute target position)
        self.target_position = np.zeros((self.Nh+1, 3)) 
        self.target_position[:,:] = self.pdes.copy() 
        self.target_position_x = self.target_position[:,0] 
        self.target_position_y = self.target_position[:,1] 
        self.target_position_z = self.target_position[:,2]

        
        # ForceEstimator

        # Sanity checks 
        if(self.config['USE_DELTA_F']):
            logger.warning("Using DELTA_F")
            try: 
                assert(self.config['FORCE_INTEGRAL'] == False)
            except: 
                logger.error("Cannot use delta_f and integral at the same time")

        if(self.config['FORCE_INTEGRAL']):
            logger.warning("Using FORCE_INTEGRAL")
            try: 
                assert(self.config['USE_DELTA_F'] == False)
            except: 
                logger.error("Cannot use delta_f and integral at the same time")
        

        self.delta_f = 0.
        
        if(self.config['USE_DELTA_TAU']):
            logger.warning("Using USE_DELTA_TAU")
            try: 
                assert(self.config['USE_DELTA_F'] == False)
                assert(self.config['FORCE_INTEGRAL'] == False)
            except: 
                logger.error("Must have USE_DELTA_F, FORCE_INTEGRAL = False")
            self.delta_tau = np.zeros(self.nv)  
            
        frame_of_interest = config['frame_of_interest']
        id_endeff = self.robot.model.getFrameId(frame_of_interest)
        self.estimator = ForceEstimator(self.robot.model, 1, 1, id_endeff, np.array(config['contacts'][0]['contactModelGains']), self.pinRef)
        
        if(self.config['USE_DELTA_TAU']):
            self.estimator = TorqueEstimator(self.robot.model, 1, id_endeff, np.array(config['contacts'][0]['contactModelGains']), self.pinRef)
        
        self.data_estimator = self.estimator.createData()


        self.tau_old = np.zeros(self.nv)


        if(self.config['USE_DELTA_TAU']):
            self.estimator.Q = 4e-3 * 0.01 * np.array([1., 1., 1., 1., 1., 1.])
            self.estimator.R = 2e-2 * 0.01 * np.ones(1)
        else:
            self.estimator.Q = 4e-3 * np.ones(self.nv)
            self.estimator.R = 2e-2 * np.ones(1)
        
        self.force_est = 0.
        self.acc_est = 0.


        # integral effect parameters
        self.force_integral = np.array([0.])
        self.KF_I = 15.
        self.alpha_f = 1. # 0.9995


        self.node_id_reach = -1
        self.node_id_contact = -1
        self.node_id_track = -1
        self.node_id_circle = -1
        self.TASK_PHASE = 0
        self.NH_SIMU   = int(self.Nh*self.dt_ocp/self.dt_ctrl)
        self.T_REACH   = int(self.config['T_REACH']/self.dt_ctrl)
        self.T_TRACK   = int(self.config['T_TRACK']/self.dt_ctrl)
        self.T_CONTACT = int(self.config['T_CONTACT']/self.dt_ctrl)
        self.T_RAMP = int(self.config['T_RAMP']/self.dt_ctrl)
        self.T_CIRCLE = int(self.config['T_CIRCLE']/self.dt_ctrl)
        logger.debug("Size of MPC horizon in simu cycles = "+str(self.NH_SIMU))
        logger.debug("Start of reaching phase in simu cycles = "+str(self.T_REACH))
        logger.debug("Start of contact phase in simu cycles = "+str(self.T_CONTACT))
        logger.debug("Start of circle phase in simu cycles = "+str(self.T_CIRCLE))
        logger.debug("OCP to SIMU time ratio = "+str(self.OCP_TO_CTRL_RATIO))

        self.t_child = 0

        self.compensation = np.zeros(3)
        self.time_df = 0.
        self.t_child, self.t_child_1 = 0, 0

        self.time = 0.
        self.time_old = 0.


    def warmup(self, thread):
        # Warm start 
        self.nb_iter = 100 
        self.u0 = pin_utils.get_tau(self.q0, self.v0, np.zeros(self.nq), self.fext0, self.robot.model, np.zeros(self.robot.model.nq))
        self.solver.xs = [self.x0 for i in range(self.Nh+1)]
        self.solver.us = [self.u0 for i in range(self.Nh)]
        self.tau_ff, self.x_des, self.K, self.t_child, self.ddp_iter, self.KKT = solveOCP(self.joint_positions[self.controlled_joint_ids], 
                                                                                        self.joint_velocities[self.controlled_joint_ids], 
                                                                                        self.solver, 
                                                                                        self.nb_iter,
                                                                                        self.target_position, 
                                                                                        self.force_weight,
                                                                                        self.TASK_PHASE,
                                                                                        self.target_force)

            
        self.nb_iter = self.config['maxiter']
    
    
    def run(self, thread):  
        
        if thread.ti == 0:
            self.T0 = time.time()        


        
        self.time = (time.time() - self.T0) * self.config['plan_freq']
              
        # # # # # # # # # 
        # Read sensors  #
        # # # # # # # # # 
        q = self.joint_positions[self.controlled_joint_ids]
        v = self.joint_velocities[self.controlled_joint_ids]
        self.a = self.joint_accelerations[self.controlled_joint_ids]

        fs = self.ft_sensor_wrench

        # When getting torque measurement from robot, do not forget to flip the sign
        if(not self.RUN_SIM):
            self.joint_torques_measured = -self.joint_torques_total  

        # Force fs is expressed in the SENSOR frame 
        # Target force in the OCP is expressed in the CENTERED CONTACT frame 
        # Transform the measurement as : SENSOR --> CONTACT --> CENTERED CONTACT 
        self.lwaMc = self.robot.data.oMf[self.contactFrameId].copy()
        self.lwaMc.translation = np.zeros(3)
        if(not self.RUN_SIM):
            f6d_sensor         = pin.Force(fs)
            # self.contact_force_6d_measured_sensor = f6d_sensor.vector.copy()
            # f6d_local          = self.cMs.act(f6d_sensor) 
            f6d_world          = self.lwaMc.act(self.cMs.act(f6d_sensor))           
        else:
            #### CAREFUL : PyBullet forces are in WORLD by default
            # f6d_local   = self.lwaMc.actInv(pin.Force(self.ft_sensor_wrench))
            f6d_world   = pin.Force(self.ft_sensor_wrench) 
        
        # if(self.pinRef == pin.LOCAL):
        #     self.contact_force_3d_measured = f6d_local.vector[:3].copy()
        # else:
        self.contact_force_3d_measured = f6d_world.linear.copy()
        
            


        alpha = 0.95
        self.force_est = alpha * self.force_est + (1-alpha) * self.contact_force_3d_measured
        self.acc_est = alpha * self.acc_est + (1-alpha) * self.a

        # # compute integral
        # self.force_integral[0] = self.alpha_f * self.force_integral[0] + (self.force_est[2] - self.target_force[0]) * self.dt_ctrl
        # self.force_integral[0] = np.core.umath.maximum(np.core.umath.minimum(self.force_integral[0], 100), -100)

        # # # # # # # # # 
        # # Update OCP  #
        # # # # # # # # 
        
        # time_to_reach   = thread.ti - self.T_REACH
        # time_to_track   = thread.ti - self.T_TRACK
        # time_to_contact = thread.ti - self.T_CONTACT
        # time_to_ramp    = thread.ti - self.T_RAMP
        # time_to_circle  = thread.ti - self.T_CIRCLE

        time_to_contact = int(self.time - self.T_CONTACT)
        time_to_ramp    = int(self.time - self.T_RAMP)
        time_to_circle  = int(self.time - self.T_CIRCLE)

        # compute integral
        if 0. <= time_to_ramp and self.config["FORCE_INTEGRAL"]:
            self.force_integral[0] = self.alpha_f * self.force_integral[0] + (self.force_est[2] - self.coef_target_force * self.target_force_traj[time_to_contact, 2]) * self.dt_ctrl
            self.force_integral[0] = np.core.umath.maximum(np.core.umath.minimum(self.force_integral[0], 100), -100)
                
        # Delta F estimation:
        
        if time_to_ramp > 0.:
            if(self.config['USE_DELTA_F'] == True):
                self.estimator.estimate(self.data_estimator, q, v, self.acc_est, self.tau_old, np.array([self.delta_f]), np.array([self.force_est[2]]))
                # Safety clipping (using np.core is 4 times faster than np.clip)
                self.delta_f = np.core.umath.maximum(np.core.umath.minimum(self.data_estimator.delta_f, self.delta_f + 1.), self.delta_f - 1.)
                self.delta_f = np.core.umath.maximum(np.core.umath.minimum(self.delta_f, 40), -40)
            elif(self.config['USE_DELTA_TAU'] == True):
                self.estimator.estimate(self.data_estimator, q, v, self.acc_est, self.tau_old, self.delta_tau, np.array([self.force_est[2]]))
                # Safety clipping (using np.core is 4 times faster than np.clip)
                self.delta_tau = np.core.umath.maximum(np.core.umath.minimum(self.data_estimator.delta_tau, self.delta_tau + 0.5), self.delta_tau - 0.5)
                self.delta_tau = np.core.umath.maximum(np.core.umath.minimum(self.delta_tau, 40), -40)
                if self.config['INTERNAL']:
                    for m in self.solver.problem.runningModels:
                        m.differential.delta_tau = - self.delta_tau
                    self.solver.problem.terminalModel.differential.delta_tau = - self.delta_tau
                    
        # Update OCP for reaching phase                   
                    
        if self.time_old <= self.T_REACH and self.T_REACH < self.time:
            print("Entering reaching phase")
            self.TASK_PHASE = 1

        if self.time_old <=self.T_TRACK and self.T_TRACK < self.time:
            print("Entering tracking phase")
            self.TASK_PHASE = 2
    
        if self.time_old <= self.T_CONTACT and self.T_CONTACT < self.time:
            # Record end-effector position at the time of the contact switch
            self.position_at_contact_switch = self.robot.data.oMf[self.contactFrameId].translation.copy()
            self.target_position[:,:] = self.position_at_contact_switch.copy()
            self.target_position_x = self.target_position[:,0] 
            self.target_position_y = self.target_position[:,1] 
            self.target_position_z = self.target_position[:,2]
            print("Entering contact phase")
            self.TASK_PHASE = 3

        if 0 <= time_to_contact:
            # set force refs over current horizon
            tf  = time_to_contact + (self.Nh+1)*self.OCP_TO_CTRL_RATIO
            self.target_force = self.coef_target_force * self.target_force_traj[time_to_contact:tf:self.OCP_TO_CTRL_RATIO, 2]
            if( self.config['USE_DELTA_F'] and self.config['INTERNAL']):
                self.target_force += self.delta_f
            if( self.config["FORCE_INTEGRAL"] and self.config['INTERNAL']):
                self.target_force -= self.KF_I * self.force_integral[0]

        if self.time_old <= self.T_CIRCLE and self.T_CIRCLE < self.time:
            self.TASK_PHASE = 4
            print("Entering circle phase")
        
        if(0 <= time_to_circle):
            # set position refs over current horizon
            tf  = time_to_circle + (self.Nh+1)*self.OCP_TO_CTRL_RATIO
            # Target in (x,y)  = circle trajectory + offset to start from current position instead of absolute target
            self.target_position[:,:2] = self.target_position_traj[time_to_circle:tf:self.OCP_TO_CTRL_RATIO,:2] + self.position_at_contact_switch[:2] - self.pdes[:2]
            # Target in z is fixed to the anchor at switch (equals absolute target if RESET_ANCHOR = False)
            # No position tracking in z : redundant with zero activation weight on z
            # self.target_position[:,2]  = self.robot.data.oMf[self.contactFrameId].translation[2].copy()
            # Record target signals
            self.target_position_x = self.target_position[:,0] 
            self.target_position_y = self.target_position[:,1] 
            self.target_position_z = self.target_position[:,2]

        # # # # # # #  
        # Solve OCP #
        # # # # # # #  
        # If planning cycle, fetch OCP solution
        if thread.ti % self.CTRL_TO_PLAN_RATIO == 0:         

            self.tau_ff, self.x_des, self.K, self.t_child, self.ddp_iter, self.KKT = solveOCP(q, v, 
                                                                self.solver,
                                                                self.nb_iter,
                                                                self.target_position, 
                                                                self.force_weight, 
                                                                self.TASK_PHASE,
                                                                self.target_force)


        # # # # # # # # 
        # Send policy #
        # # # # # # # #     

        # Riccati policy (optional) on (q,v) 
        # if(self.config['RICCATI']):
        #     self.tau_riccati = self.K[:,:self.nq+self.nv] @ (self.x_des[:self.nq+self.nv] - np.concatenate([q, v]))
        #     self.tau  = self.tau_ff + self.tau_riccati
        # else:
        self.tau = self.tau_ff
        
        Jac = pin.computeFrameJacobian(self.robot.model, self.robot.data, q, self.contactFrameId, pin.LOCAL_WORLD_ALIGNED)[:3, self.controlled_joint_ids]
        # Mismatch correction as feedforward term
        if( self.config['USE_DELTA_F'] and 0 <= time_to_contact and self.config['INTERNAL'] == False ):
                        # self.tau -= Jac.T @ np.array([0., 0., float(self.delta_f)]) 
            self.tau -= Jac[2] * self.delta_f 
            
            
            
        if(self.config['USE_DELTA_TAU'] and self.config['INTERNAL'] == False and 0 <= time_to_contact):
            self.tau += self.delta_tau 

        if( self.config["FORCE_INTEGRAL"] and 0 <= time_to_contact and self.config['INTERNAL'] == False ):
            self.tau += self.KF_I * Jac.T @ np.array([0., 0., self.force_integral[0]])
            
            
        # Save old torque as estimator is not aware of the lateral force model
        self.tau_old = self.tau.copy()
        
        
        # Lateral force model
        if( self.config['USE_LATERAL_FRICTION'] and 0 <= time_to_contact ):
            self.tau -= Jac.T @ np.array([self.force_est[0], self.force_est[1], 0.])
            
        if( self.config['USE_COULOMB'] and 0 <= time_to_contact ):
            # Jac = pin.computeFrameJacobian(self.robot.model, self.robot.data, q, self.contactFrameId, pin.LOCAL_WORLD_ALIGNED)[:3, self.controlled_joint_ids]
            pin.forwardKinematics(self.robot.model, self.robot.data, q, v)
            v_ee    = pin.getFrameVelocity(self.robot.model, self.robot.data, self.contactFrameId, pin.LOCAL_WORLD_ALIGNED).linear
            v_ee[2] = 0.
            norm_ = np.linalg.norm(v_ee)
            if(norm_ > 0):
                self.compensation = -0.35 * self.force_est[2] * np.tanh(10. * norm_) * v_ee / norm_ / np.sqrt(2)
                self.tau -= Jac.T @ self.compensation
                

            
        # Compute gravity
        self.tau_gravity = pin.rnea(self.robot.model, self.robot.data, self.joint_positions[self.controlled_joint_ids], np.zeros(self.nv), np.zeros(self.nv))

        # Substract gravity for real robot
        if(self.RUN_SIM == False):
            self.tau -= self.tau_gravity

        if(not self.RUN_SIM and len(self.fixed_ids) > 0):
            self.tau_PD   = -self.gain_P * self.joint_positions[self.fixed_ids] - self.gain_D * self.joint_velocities[self.fixed_ids]
            self.tau_full = np.zeros(7)
            self.tau_full[self.controlled_joint_ids] = self.tau.copy()
            self.tau_full[self.fixed_ids] = self.tau_PD.copy()
            self.tau = self.tau_full.copy()
            
        ###### DANGER SEND ONLY GRAV COMP
        # self.tau = np.zeros_like(self.tau_full)
        
        self.head.set_control('ctrl_joint_torques', self.tau)     
        
        
        pin.framesForwardKinematics(self.robot.model, self.robot.data, q)
        
        self.t_run = time.time() - self.T0 - self.time / self.config['ctrl_freq']
        self.time_old = self.time