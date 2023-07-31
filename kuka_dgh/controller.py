import numpy as np
import pinocchio as pin 

import time
import matplotlib.pyplot as plt 

from classical_mpc.ocp import OptimalControlProblemClassical
from core_mpc import pin_utils
from core_mpc.misc_utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

from multiprocessing import Pipe, Process
from robot_properties_kuka.config import IiwaConfig

from force_observer import ForceEstimator, MHForceEstimator


NO_PIPE = True



def solveOCP(q, v, ddp, nb_iter, node_id_reach, target_reach, node_id_contact, node_id_track, node_id_circle, force_weight, TASK_PHASE, target_force, target_velocity):
        t = time.time()
        # Update initial state + warm-start
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
        # Update OCP for "increase weights" phase
        if(TASK_PHASE == 2):
            # If node id is valid
            if(node_id_track <= ddp.problem.T and node_id_track >= 0):
                # Updates nodes between node_id and terminal node 
                for k in range( node_id_track, ddp.problem.T+1, 1 ):
                    w = min(2.*(k + 1. - node_id_track) , 5)
                    m[k].differential.costs.costs["translation"].active = True
                    m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                    m[k].differential.costs.costs["translation"].weight = w
        # Update OCP for contact phase
        if(TASK_PHASE == 3):
            # If node id is valid
            if(node_id_contact <= ddp.problem.T and node_id_contact >= 0):
                # Updates nodes between node_id and terminal node 
                for k in range( node_id_contact, ddp.problem.T+1, 1 ):  
                    m[k].differential.costs.costs["translation"].active = True
                    m[k].differential.costs.costs["translation"].cost.residual.reference = target_reach[k]
                    m[k].differential.costs.costs["translation"].weight = 2.
                    m[k].differential.costs.costs['rotation'].active = True
                    m[k].differential.costs.costs['rotation'].cost.residual.reference = pin.utils.rpyToMatrix(np.pi, 0, np.pi)
                    # activate contact and force cost
                    m[k].differential.contacts.changeContactStatus("contact", True)
                    if(k!=ddp.problem.T):
                        fref = pin.Force(np.array([0., 0., target_force[k], 0., 0., 0.]))
                        m[k].differential.costs.costs["force"].active = True
                        m[k].differential.costs.costs["force"].weight = force_weight
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
                    m[k].differential.costs.costs["translation"].weight = 60. 
                    m[k].differential.costs.costs['rotation'].active = True
                    m[k].differential.costs.costs['rotation'].cost.residual.reference = pin.utils.rpyToMatrix(np.pi, 0, np.pi)
                    # activate contact and force cost
                    m[k].differential.contacts.changeContactStatus("contact", True)
                    if(k!=ddp.problem.T):
                        fref = pin.Force(np.array([0., 0., target_force[k], 0., 0., 0.]))
                        m[k].differential.costs.costs["force"].active = True
                        m[k].differential.costs.costs["force"].weight = force_weight
                        m[k].differential.costs.costs["force"].cost.residual.reference = fref
        # get predicted force from rigid model (careful : expressed in LOCAL !!!)
        jf = ddp.problem.runningDatas[0].differential.multibody.contacts.contacts['contact'].f
        jMf = ddp.problem.runningDatas[0].differential.multibody.contacts.contacts['contact'].jMf
        fpred = jMf.actInv(jf).linear
        problem_formulation_time = time.time()
        t_child_1 =  problem_formulation_time - t
        # Solve OCP 
        ddp.solve(xs_init, us_init, maxiter=nb_iter, isFeasible=False)
        solve_time = time.time()
        ddp_iter = ddp.iter
        t_child =  solve_time - problem_formulation_time
        # Send solution to parent process + riccati gains
        return ddp.us, ddp.xs, ddp.K, fpred, t_child, ddp_iter, t_child_1



class ClassicalMPCContact:

    def __init__(self, head, robot, config, f0, contact_placement, cMs, run_sim, controlled_joints_names=["A1", "A2", "A3", "A4", "A5", "A6", "A7"]):
        """
        Input:
            head              : thread head
            robot_model       : pinocchio model
            config            : MPC config yaml file
            f0                : initial measured force
            contact_placement : placement of the contact surface
            cMs               : placement of the sensor frame w.r.t. contact frame
            run_sim           : boolean sim or real
            controlled_joint_names : which joints are controlled by the MPC
        """
        self.robot   = robot
        self.head    = head
        self.RUN_SIM = run_sim
        self.joint_positions  = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor("joint_velocities")
        self.joint_accelerations = head.get_sensor("joint_accelerations")
        if not self.RUN_SIM:
            self.joint_ext_torques = head.get_sensor("joint_torques_external")
            self.joint_cmd_torques = head.get_sensor("joint_torques_commanded")
            self.ft_sensor_wrench  = head.get_sensor("ft_sensor_wrench") # in sensor frame !!
        else: 
            self.ft_sensor_wrench = self.head._sensor__force_plate_force[0]

        if(NO_PIPE == False):
            self.parent_conn, self.child_conn = Pipe()
            self.sent = False

        self.nq = self.robot.model.nq
        self.nv = self.robot.model.nv

        logger.warning("Controlled model dimensions : ")
        logger.warning(" nq = "+str(self.nq))
        logger.warning(" nv = "+str(self.nv))
        
        full_robot = IiwaConfig.buildRobotWrapper()
        full_model = full_robot.model
        self.controlled_joint_ids = [full_model.joints[full_model.getJointId(joint_name)].idx_q for joint_name in controlled_joints_names] 
        logger.warning("Controlled names = \n"+str(controlled_joints_names))
        logger.warning("Controlled joint ids = \n"+str(self.controlled_joint_ids))
        self.fixed_ids = []
        for i in range(7):
            if(i not in self.controlled_joint_ids):
                self.fixed_ids.append(i)
        logger.warning("Fixed joint ids = \n"+str(self.fixed_ids))
        self.gain_P = 100.
        self.gain_D = 30.
        
        # Get config + initial state (from sim or sensors)
        self.config = config
        if(self.RUN_SIM):
            self.controlled_joint_ids = range(len(self.controlled_joint_ids)) 
            self.q0 = np.asarray(config['q0'])[self.controlled_joint_ids]    
            self.v0 = self.joint_velocities[self.controlled_joint_ids]    
        else:
            self.q0 = self.joint_positions[self.controlled_joint_ids]    
            self.v0 = self.joint_velocities[self.controlled_joint_ids]  
        self.x0 = np.concatenate([self.q0, self.v0]) 
        self.cMs = cMs
        self.sensorFrameId = self.robot.model.getFrameId('ft_sensor')
        self.contactFrameId = self.robot.model.getFrameId('contact')
        logger.warning("Sensor frame id = "+str(self.sensorFrameId))
        logger.warning("Contact frame id = "+str(self.contactFrameId))
        pin.framesForwardKinematics(self.robot.model, self.robot.data, self.q0)
        self.Nh = self.config['N_h']
        self.dt_ocp  = self.config['dt']
        self.dt_plan = 1./self.config['plan_freq']
        self.dt_simu = 1./self.config['simu_freq']
        self.ocp_to_sim_ratio = 1. / ( self.config['simu_freq'] * self.dt_ocp )
        self.sim_to_plan_ratio = self.config['simu_freq']/self.config['plan_freq']
        # Create OCP
        self.oMc = contact_placement
        self.fext0 = pin_utils.get_external_joint_torques(contact_placement, f0, robot)  
        # logger.warning(self.fext0)
        self.ddp = OptimalControlProblemClassical(robot, self.config).initialize(self.x0, callbacks=False)
        # !!! Deactivate all costs & contact models initially !!!
        models = list(self.ddp.problem.runningModels) + [self.ddp.problem.terminalModel]
        for k,m in enumerate(models):
            # logger.debug(str(m.differential.costs.active.tolist()))
            m.differential.costs.costs["translation"].active = False
            m.differential.contacts.changeContactStatus("contact", False)
            m.differential.costs.costs['rotation'].active = False
            m.differential.costs.costs['rotation'].cost.residual.reference = pin.utils.rpyToMatrix(np.pi, 0., np.pi)
            
        # Allocate MPC data
        self.us = self.ddp.us ; self.xs = self.ddp.xs ; self.Ks = self.ddp.K 
        self.x = self.xs[0] ; self.tau_ff = self.us[0] ; self.K = self.Ks[0]
        self.tau = self.tau_ff.copy() ; self.tau_riccati = np.zeros(self.tau.shape)
        self.tau_ext = self.fext0.copy()
        self.x1 = self.xs[1]
        self.nb_ctrl = 0
        self.nb_plan = 0
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
            self.contact_force_6d_measured_sensor = f6d_sensor.vector.copy()
            f6d_local   = self.cMs.act(f6d_sensor)
            f6d_world   = self.lwaMc.act(self.cMs.act(f6d_sensor))  
        else:
            #### CAREFUL : PyBullet forces are in WORLD by default...
            f6d_local   = self.lwaMc.actInv(pin.Force(self.ft_sensor_wrench))
            f6d_world   = pin.Force(self.ft_sensor_wrench) 
        if(self.config['pinRefFrame'] == 'LOCAL'):
            self.pinRef = pin.LOCAL
            logger.warning("LOCAL contact force !")
            self.contact_force_3d_measured = f6d_local.vector[:3].copy()
            self.coef_target_force = -1.
        else:
            self.pinRef = pin.LOCAL_WORLD_ALIGNED
            logger.warning("LOCAL_WORLD_ALIGNED contact force !")
            self.contact_force_3d_measured  = f6d_world.vector[:3].copy()
            self.coef_target_force = +1.  
             
        # Task phases management & cost parameters
        F_MIN = 5.
        F_MAX = self.config['frameForceRef'][2]
        N_total = int((self.config['T_tot'] - self.config['T_CONTACT'])/self.dt_ocp + self.Nh)

        N_ramp = int((self.config['T_RAMP'] - self.config['T_CONTACT']) / self.dt_ocp)
        self.target_force_traj = np.zeros((N_total, 3))
        self.target_force_traj[:N_ramp, 2] = [F_MIN + (F_MAX - F_MIN)*i/N_ramp for i in range(N_ramp)]
        self.target_force_traj[N_ramp:, 2] = F_MAX
        # self.target_force_traj[N_sinus*self.Nh:, 2] = [F_MAX + 20.*np.round(np.sin(0.01 * (2*np.pi/self.Nh) * i/2 )  ) for i in range(N_total-N_sinus*self.Nh)]
        # plt.plot(self.target_force_traj)
        # plt.show()
        self.target_force = np.zeros(self.Nh+1)
        self.force_weight = self.config['frameForceWeight']

        # Circle trajectory 
        N_total_pos = int((self.config['T_tot'] - self.config['T_REACH'])/self.dt_ocp + self.Nh)
        N_circle = int((self.config['T_tot'] - self.config['T_CIRCLE'])/self.dt_ocp) + self.Nh
        self.target_position_traj = np.zeros( (N_total_pos, 3) )
        self.target_velocity_traj = np.zeros( (N_total_pos, 3) )
        # absolute desired position
        self.oPc_offset = np.asarray(self.config['oPc_offset'])
        self.pdes = np.asarray(self.config['contactPosition']) + self.oPc_offset
        radius = 0.07 ; omega = 3.
        # radius = 0.0 ; omega = 3.

        self.target_position_traj[0:N_circle, :] = [np.array([self.pdes[0] + radius * (1-np.cos(i*self.dt_ocp*omega)), 
                                                              self.pdes[1] - radius * np.sin(i*self.dt_ocp*omega),
                                                              self.pdes[2]]) for i in range(N_circle)]
        self.target_velocity_traj[0:N_circle, :] = [np.array([radius * omega * np.cos(i*self.dt_ocp*omega), 
                                                              radius * omega * np.sin(i*self.dt_ocp*omega),
                                                              0.]) for i in range(N_circle)]
        self.target_position_traj[N_circle:, :] = self.target_position_traj[N_circle-1,:]
        self.target_velocity_traj[N_circle:, :] = np.zeros(3)
        # plt.plot(self.target_position_traj, label='pos')
        # plt.plot(self.target_velocity_traj, label='vel')
        # plt.show()
        # Targets over one horizon (initially = absolute target position)
        self.target_position = np.zeros((self.Nh+1, 3)) 
        self.target_velocity = np.zeros((self.Nh+1, 3)) 
        self.target_position[:,:] = self.pdes.copy() 
        self.target_velocity[:,:] = np.zeros(3) 
        self.target_position_x = self.target_position[:,0] 
        self.target_position_y = self.target_position[:,1] 
        self.target_position_z = self.target_position[:,2]
        self.target_velocity_x = self.target_velocity[:,0] 
        self.target_velocity_y = self.target_velocity[:,1] 
        self.target_velocity_z = self.target_velocity[:,2]

        
        # ForceEstimator
        frame_of_interest = config['frame_of_interest']
        id_endeff = robot.model.getFrameId(frame_of_interest)
        if config['DF_HORIZON'] == 0:
            self.estimator = ForceEstimator(self.robot.model, 1, 1, id_endeff, np.array(config['contacts'][0]['contactModelGains']), self.pinRef)
        else:
            self.estimator = MHForceEstimator(config['DF_HORIZON'], self.robot.model, 1, id_endeff, np.array(config['contacts'][0]['contactModelGains']), self.pinRef)

        self.data_estimator = self.estimator.createData()
        self.delta_f = 0.
        self.estimator.Q = 4e-3 * np.ones(7)
        self.estimator.R = 2e-2 * np.ones(1)

        self.buffer_length = max(1, config['DF_HORIZON'])
        self.buffer_q   = np.zeros(self.buffer_length * self.nv)
        self.buffer_v   = np.zeros(self.buffer_length * self.nv)
        self.buffer_a   = np.zeros(self.buffer_length * self.nv)
        self.buffer_tau = np.zeros(self.buffer_length * self.nv)
        self.buffer_f   = np.zeros(self.buffer_length)
        self.tau_old = np.zeros(self.nv)


        self.node_id_reach = -1
        self.node_id_contact = -1
        self.node_id_track = -1
        self.node_id_circle = -1
        self.TASK_PHASE = 0
        self.NH_SIMU   = int(self.Nh*self.dt_ocp/self.dt_simu)
        self.T_REACH   = int(self.config['T_REACH']/self.dt_simu)
        self.T_TRACK   = int(self.config['T_TRACK']/self.dt_simu)
        self.T_CONTACT = int(self.config['T_CONTACT']/self.dt_simu)
        self.T_RAMP = int(self.config['T_RAMP']/self.dt_simu)
        self.T_CIRCLE = int(self.config['T_CIRCLE']/self.dt_simu)
        self.OCP_TO_SIMU_CYCLES = 1./(self.dt_simu / self.dt_ocp)
        logger.debug("Size of MPC horizon in simu cycles = "+str(self.NH_SIMU))
        logger.debug("Start of reaching phase in simu cycles = "+str(self.T_REACH))
        logger.debug("Start of contact phase in simu cycles = "+str(self.T_CONTACT))
        logger.debug("Start of circle phase in simu cycles = "+str(self.T_CIRCLE))
        logger.debug("OCP to SIMU time ratio = "+str(self.OCP_TO_SIMU_CYCLES))

 

    def warmup(self, thread):
        # Warm start 
        self.nb_iter = 100 
        self.u0 = pin_utils.get_tau(self.q0, self.v0, np.zeros(self.nq), self.fext0, self.robot.model, np.zeros(self.robot.model.nq))
        self.ddp.xs = [self.x0 for i in range(self.Nh+1)]
        self.ddp.us = [self.u0 for i in range(self.Nh)]
        self.is_plan_updated = False
        # No pipe
        if(NO_PIPE):
            self.us, self.xs, self.Ks, self.fpred, self.t_child, self.ddp_iter, self.t_child_1 = solveOCP(self.joint_positions[self.controlled_joint_ids], 
                                                                                            self.joint_velocities[self.controlled_joint_ids], 
                                                                                            self.ddp, 
                                                                                            self.nb_iter,
                                                                                            self.node_id_reach, 
                                                                                            self.target_position, 
                                                                                            self.node_id_contact, 
                                                                                            self.node_id_track,
                                                                                            self.node_id_circle, 
                                                                                            self.force_weight,
                                                                                            self.TASK_PHASE,
                                                                                            self.target_force, 
                                                                                            self.target_velocity)

        # With pipe
        else:
            self.subp = Process(target=rt_SolveOCP, args=(self.child_conn, 
                                                        self.ddp, 
                                                        self.nb_iter,
                                                        self.node_id_reach, 
                                                        self.target_position,
                                                        self.node_id_contact, 
                                                        self.node_id_track,
                                                        self.node_id_circle,
                                                        self.force_weight,
                                                        self.TASK_PHASE,
                                                        self.target_force,
                                                        self.target_velocity))
            self.subp.start()
            # Read sensors and publish real state 
            self.parent_conn.send((self.joint_positions[self.controlled_joint_ids], 
                                self.joint_velocities[self.controlled_joint_ids], 
                                self.node_id_reach, 
                                self.target_position, 
                                self.node_id_contact, 
                                self.node_id_track,
                                self.node_id_circle,
                                self.force_weight,
                                self.TASK_PHASE,
                                self.target_force,
                                self.target_velocity,
                                self.nb_iter))
            self.us, self.xs, self.Ks, self.fpred, self.t_child, self.ddp_iter, self.t_child_1  = self.parent_conn.recv()

        if(self.pinRef != pin.LOCAL):
            self.fpred = self.lwaMc.rotation @ self.fpred
            
        self.check = 0
        self.nb_iter = self.config['maxiter']
        self.count = 0
        self.sent = False
    

    def run(self, thread):  
        t1 = time.time()
              
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
            self.contact_force_6d_measured_sensor = f6d_sensor.vector.copy()
            f6d_local          = self.cMs.act(f6d_sensor) 
            f6d_world          = self.lwaMc.act(self.cMs.act(f6d_sensor))           
        else:
            #### CAREFUL : PyBullet forces are in WORLD by default
            f6d_local   = self.lwaMc.actInv(pin.Force(self.ft_sensor_wrench))
            f6d_world   = pin.Force(self.ft_sensor_wrench) 
        
        if(self.pinRef == pin.LOCAL):
            self.contact_force_3d_measured = f6d_local.vector[:3].copy()
        else:
            self.contact_force_3d_measured = f6d_world.linear.copy()
        

        # # # # # # # # # 
        # # Update OCP  #
        # # # # # # # # # 
        time_to_reach   = int(thread.ti - self.T_REACH)
        time_to_track   = int(thread.ti - self.T_TRACK)
        time_to_contact = int(thread.ti - self.T_CONTACT)
        time_to_ramp    = int(thread.ti - self.T_RAMP)
        time_to_circle  = int(thread.ti - self.T_CIRCLE)



        # Delta F estimation:
        

        self.buffer_q[self.nv:]   = self.buffer_q[:-self.nv]
        self.buffer_v[self.nv:]   = self.buffer_v[:-self.nv]
        self.buffer_a[self.nv:]   = self.buffer_a[:-self.nv]
        self.buffer_tau[self.nv:]   = self.buffer_tau[:-self.nv]
        self.buffer_f[1:]   = self.buffer_f[:-1]



        self.buffer_q[:self.nv]   = q
        self.buffer_v[:self.nv]   = v
        self.buffer_a[:self.nv]   = self.a
        self.buffer_tau[:self.nv] = self.tau_old
        self.buffer_f[:1]         = np.array([self.contact_force_3d_measured[2]])

        t0 = time.time()
        if time_to_ramp > 0:
            self.estimator.estimate(self.data_estimator, self.buffer_q, self.buffer_v, self.buffer_a, self.buffer_tau, np.array([self.delta_f]), self.buffer_f)
            self.delta_f = np.clip(self.data_estimator.delta_f, self.delta_f - 0.2, self.delta_f + 0.2)
            # Safety clipping
            self.delta_f = np.clip(self.delta_f, -40, 40)
        self.time_df = time.time() - t0


        if(time_to_reach == 0): 
            print("Entering reaching phase")
        # If tracking phase enters the MPC horizon, start updating models from the end with tracking models      
        if(0 <= time_to_reach and time_to_reach <= self.NH_SIMU):
            self.TASK_PHASE = 1
            # If current time matches an OCP node 
            if(int(self.T_REACH - thread.ti)%self.OCP_TO_SIMU_CYCLES == 0):
                # Select IAM
                self.node_id_reach = self.Nh + int((self.T_REACH - thread.ti)/self.OCP_TO_SIMU_CYCLES)

        if(time_to_track == 0): 
            print("Entering tracking phase")
        # If "increase weights" phase enters the MPC horizon, start updating models from the end with tracking models      
        if(0 <= time_to_track and time_to_track <= self.NH_SIMU):
            self.TASK_PHASE = 2
            # If current time matches an OCP node 
            if(int(self.T_TRACK - thread.ti)%self.OCP_TO_SIMU_CYCLES == 0):
                # Select IAM
                self.node_id_track = self.Nh + int((self.T_TRACK - thread.ti)/self.OCP_TO_SIMU_CYCLES)

        if(time_to_contact == 0): 
            # Record end-effector position at the time of the contact switch
            self.position_at_contact_switch = self.robot.data.oMf[self.contactFrameId].translation.copy()
            self.target_position[:,:] = self.position_at_contact_switch.copy()
            self.target_position_x = self.target_position[:,0] 
            self.target_position_y = self.target_position[:,1] 
            self.target_position_z = self.target_position[:,2]
            print("Entering contact phase")
            # self.target_position = self.robot.data.oMf[self.frameId].translation
        # If contact phase enters horizon start updating models from the the end with contact models
        if(0 <= time_to_contact and time_to_contact <= self.NH_SIMU):
            self.TASK_PHASE = 3
            # If current time matches an OCP node 
            if(int(self.T_CONTACT - thread.ti)%self.OCP_TO_SIMU_CYCLES == 0):
                # Select IAM
                self.node_id_contact = self.Nh + int((self.T_CONTACT - thread.ti)/self.OCP_TO_SIMU_CYCLES)

        if(0 <= time_to_contact and time_to_contact%self.OCP_TO_SIMU_CYCLES == 0):
            # set force refs over current horizon
            ti  = int((thread.ti - self.T_CONTACT)/self.OCP_TO_SIMU_CYCLES)
            tf  = ti + self.Nh+1
            self.target_force = self.coef_target_force * self.target_force_traj[ti:tf, 2]
            if self.config['USE_DELTA_F']:
                self.target_force += self.delta_f


        if(time_to_circle == 0): 
            print("Entering circle phase")
        # If circle tracking phase enters the MPC horizon, start updating models from the end with tracking models      
        if(0 <= time_to_circle and time_to_circle <= self.NH_SIMU):
            self.TASK_PHASE = 4
            # If current time matches an OCP node 
            if(time_to_circle%self.OCP_TO_SIMU_CYCLES == 0):
                # Select IAM
                self.node_id_circle = self.Nh - int(time_to_circle/self.OCP_TO_SIMU_CYCLES)

        if(0 <= time_to_circle and time_to_circle%self.OCP_TO_SIMU_CYCLES == 0):
            # set position refs over current horizon
            ti  = int(time_to_circle/self.OCP_TO_SIMU_CYCLES)
            tf  = ti + self.Nh+1
            # Target in (x,y)  = circle trajectory + offset to start from current position instead of absolute target
            offset_xy = self.position_at_contact_switch[:2] - self.pdes[:2]
            self.target_position[:,:2] = self.target_position_traj[ti:tf,:2] + offset_xy
            # Target in z is fixed to the anchor at switch (equals absolute target if RESET_ANCHOR = False)
            # No position tracking in z : redundant with zero activation weight on z
            self.target_position[:,2]  = self.robot.data.oMf[self.contactFrameId].translation[2].copy()
            # Record target signals
            self.target_position_x = self.target_position[:,0] 
            self.target_position_y = self.target_position[:,1] 
            self.target_position_z = self.target_position[:,2]
            # Record target signals                
            self.target_velocity[:,:2] = self.target_velocity_traj[ti:tf,:2] 
            self.target_velocity[:,2]  = 0.
            self.target_velocity_x = self.target_velocity[:,0] 
            self.target_velocity_y = self.target_velocity[:,1] 
            self.target_velocity_z = self.target_velocity[:,2]

        # # # # # # #  
        # Solve OCP #
        # # # # # # #  
        # If planning cycle, fetch OCP solution
        self.t_child, self.t_child_1 = 0, 0
        if thread.ti % int(self.sim_to_plan_ratio) == 0:         
            # No pipe
            if(NO_PIPE):
                self.count = 0
                self.us, self.xs, self.Ks, self.fpred, self.t_child, self.ddp_iter, self.t_child_1 = solveOCP(q, v, 
                                                                    self.ddp,
                                                                    self.nb_iter,
                                                                    self.node_id_reach, 
                                                                    self.target_position, 
                                                                    self.node_id_contact, 
                                                                    self.node_id_track, 
                                                                    self.node_id_circle,
                                                                    self.force_weight, 
                                                                    self.TASK_PHASE,
                                                                    self.target_force, 
                                                                    self.target_velocity)
            # With pipe
            else:
                if thread.ti != 0 and not self.sent:
                    self.is_plan_updated = False
                    self.parent_conn.send((q, v, 
                                                self.node_id_reach, 
                                                self.target_position, 
                                                self.node_id_contact, 
                                                self.node_id_track,
                                                self.node_id_circle,
                                                self.force_weight,
                                                self.TASK_PHASE,
                                                self.target_force,
                                                self.target_velocity, 
                                                self.nb_iter)) 
                    self.sent = True

            if(self.pinRef != pin.LOCAL):
                self.fpred = self.lwaMc.rotation @ self.fpred
                
        if not NO_PIPE and self.parent_conn.poll() and not self.is_plan_updated:
            self.count = 0
            self.us, self.xs, self.Ks, self.fpred, self.t_child, self.ddp_iter, self.t_child_1 = self.parent_conn.recv()
            # record predictions here if necessary 
            self.sent = False

            # Increment planning counter
            self.nb_plan += 1
            self.is_plan_updated = True


        # # # # # # # # 
        # Send policy #
        # # # # # # # #
        # Linear interpolation of torque control input & desired state 
        ctr_index = int(self.count*self.ocp_to_sim_ratio)
        if ctr_index > self.Nh-1:
            self.tau_ff   = self.us[-1]     
            self.x_des    = self.xs[-1]  
            K = self.Ks[-1]
            print("DANGER")
        else:
            self.tau_ff   = self.us[ctr_index]     
            self.x_des    = self.xs[ctr_index+1]  
            K = self.Ks[ctr_index]

        # Riccati policy (optional) on (q,v) 
        if(self.config['RICCATI']):
            self.tau_riccati = K[:,:self.nq+self.nv] @ (self.x_des[:self.nq+self.nv] - np.concatenate([q, v]))
            self.tau  = self.tau_ff + self.tau_riccati
        else:
            self.tau = self.tau_ff
            
        self.count += 1

        self.tau_old = self.tau.copy()

        if( self.config['USE_LATERAL_FRICTION'] and 0 <= time_to_contact):
            Jac = pin.computeFrameJacobian(self.robot.model, self.robot.data, q, self.contactFrameId, pin.LOCAL_WORLD_ALIGNED)[:3, self.controlled_joint_ids]
            self.tau -= Jac.T @ np.array([self.contact_force_3d_measured[0], self.contact_force_3d_measured[1], 0])


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

        self.nb_ctrl += 1
        
        self.t_run = time.time() - t1
