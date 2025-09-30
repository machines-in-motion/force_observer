
import numpy as np
import pinocchio as pin
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib
from croco_mpc_utils import pinocchio_utils as pin_utils

import pybullet as p

from croco_mpc_utils.ocp_data import MPCDataHandlerClassical
from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger




class MPCDataHandlerClassicalWithEstimator(MPCDataHandlerClassical):

  def __init__(self, config, robot):
    super().__init__(config, robot)
 
  def init_sim_data(self, x0):
      super().init_sim_data(x0)
      self.init_estimates()


  def init_estimates(self):
    '''
    Allocate data for simulation state & force estimates 
    '''
    self.delta_f_SIMU = np.zeros((self.N_simu+1, 3))


  def record_simu_cycle_estimates(self, nb_simu, delta_f):
    '''
    Records the estimate of the force error
     Input:
      nb_simu  : simulation cycle number
      delta_f  : delta_f 
    '''
    self.delta_f_SIMU[nb_simu+1, :] = delta_f


  # Extract MPC simu-specific plotting data from sim data
  def extract_data(self, frame_of_interest):
    '''
    Extract plot data from simu data
    '''
    plot_data = super().extract_data(frame_of_interest)
    # Get costs
    plot_data['delta_f_SIMU'] = self.delta_f_SIMU
    return plot_data

  def plot_mpc_delta_f(self, plot_data, PLOT_PREDICTIONS=False, pred_plot_sampling=100, SAVE=False, SAVE_DIR=None, SAVE_NAME=None, SHOW=True, AUTOSCALE=False):
      '''
      Plot delta_f
      Input:
        plot_data                 : plotting data
        PLOT_PREDICTIONS          : True or False
        pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                    to avoid huge amount of plotted data 
                                    ("1" = plot all)
        SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
        SHOW                      : show plots
        AUTOSCALE                 : rescale y-axis of endeff plot 
                                    based on maximum value taken
      '''
      logger.info('Plotting delta_f data...')
      T_tot = plot_data['T_tot']
      N_simu = plot_data['N_simu']
      dt_simu = plot_data['dt_simu']
      # Create time spans for X and U + Create figs and subplots
      t_span_simu = np.linspace(0, T_tot - dt_simu, N_simu)
      fig, ax = plt.subplots(3, 2, figsize=(19.2,10.8), sharex='col') 
      # Plot endeff
      xyz = ['x', 'y', 'z']
      for i in range(3):
          # EE linear force
          ax[i,0].plot(t_span_simu, plot_data['delta_f_SIMU'][:N_simu,i], color='b', linestyle='-', marker='.', label='delta_f', alpha=1.)
          ax[i,0].set_ylabel('$\\lambda^{EE}_%s$  (N)'%xyz[i], fontsize=16)
          ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
          ax[i,0].grid(True)
          # EE angular force 
          ax[i,1].plot(t_span_simu, plot_data['delta_f_SIMU'][:N_simu,i]*0., color='b', linestyle='-', marker='.', label='delta_f', alpha=1.)
          ax[i,1].set_ylabel('$\\tau^{EE}_%s$  (Nm)'%xyz[i], fontsize=16)
          ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
          ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
          ax[i,1].grid(True)
      
      # Align
      fig.align_ylabels(ax[:,0])
      fig.align_ylabels(ax[:,1])
      ax[i,0].set_xlabel('t (s)', fontsize=16)
      ax[i,1].set_xlabel('t (s)', fontsize=16)
      # Set ylim if any
      TOL = 1e-3
      if(AUTOSCALE):
          for i in range(3):
              ax[i,0].set_ylim(-50, 50) 
              ax[i,1].set_ylim(-50, 50) 

      handles_p, labels_p = ax[0,0].get_legend_handles_labels()
      fig.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})
      # Titles
      fig.suptitle('End-effector forces', size=18)
      # Save figs
      if(SAVE):
          figs = {'f': fig}
          if(SAVE_DIR is None):
              logger.error("Please specify SAVE_DIR")
          if(SAVE_NAME is None):
              SAVE_NAME = 'testfig'
          for name, fig in figs.items():
              fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
      if(SHOW):
          plt.show() 
      return fig, ax

  def plot_mpc_force(self, plot_data, PLOT_PREDICTIONS=False, pred_plot_sampling=100, SAVE=False, SAVE_DIR=None, SAVE_NAME=None, SHOW=True, AUTOSCALE=False):
    fig, ax = super().plot_mpc_force(plot_data, PLOT_PREDICTIONS, pred_plot_sampling, SAVE, SAVE_DIR, SAVE_NAME, False, AUTOSCALE)
    # Add delta f in ax
    T_tot = plot_data['T_tot']
    N_simu = plot_data['N_simu']
    dt_simu = plot_data['dt_simu']
    t_span_simu = np.linspace(0, T_tot - dt_simu, N_simu)
    for i in range(3):
        ax[i,0].plot(t_span_simu, plot_data['force_mea_SIMU'][:N_simu,i] + plot_data['delta_f_SIMU'][:N_simu,i], color='y', linestyle='-', marker='.', label='f + delta_f', alpha=0.5)
    if(SHOW):
        plt.show() 
    return fig, ax

  def plot_mpc_results(self, plot_data, which_plots=None, PLOT_PREDICTIONS=False, 
                                                pred_plot_sampling=100, 
                                                SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                                                SHOW=True,
                                                AUTOSCALE=False):
      '''
      Plot sim data
      Input:
        plot_data                 : plotting data
        PLOT_PREDICTIONS          : True or False
        pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                    to avoid huge amount of plotted data 
                                    ("1" = plot all)
        SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
        SHOW                      : show plots
        AUTOSCALE                 : rescale y-axis of endeff plot 
                                    based on maximum value taken
      '''

      plots = {}
      super().plot_mpc_results(plot_data, which_plots, PLOT_PREDICTIONS, 
                                                pred_plot_sampling, 
                                                SAVE, SAVE_DIR, SAVE_NAME,
                                                False,
                                                AUTOSCALE)
      if('delta_f' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
          plots['delta_f'] = self.plot_mpc_delta_f(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                              pred_plot_sampling=pred_plot_sampling, 
                                              SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                              SHOW=False, AUTOSCALE=AUTOSCALE)
      if(SHOW):
          plt.show() 
      plt.close('all')


# Display
def display_ball(p_des, RADIUS=.05, COLOR=[1.,1.,1.,1.]):
    '''
    Create a sphere visual object in PyBullet (no collision)
    Transformed because reference p_des is in pinocchio WORLD frame, which is different
    than PyBullet WORLD frame if the base placement in the simulator is not (eye(3), zeros(3))
    INPUT: 
        p_des           : desired position of the ball in pinocchio.WORLD
        robot_base_pose : initial pose of the robot BASE in bullet.WORLD
        RADIUS          : radius of the ball
        COLOR           : color of the ball
    '''
    # pose of the sphere in bullet WORLD
    M = pin.SE3(np.eye(3), p_des)  # ok for talos reduced since pin.W = bullet.W but careful with talos_arm if base is moved
    quat = pin.SE3ToXYZQUAT(M)     
    visualBallId = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                       radius=RADIUS,
                                       rgbaColor=COLOR,
                                       visualFramePosition=quat[:3],
                                       visualFrameOrientation=quat[3:])
    ballId = p.createMultiBody(baseMass=0.,
                               baseInertialFramePosition=[0.,0.,0.],
                               baseVisualShapeIndex=visualBallId,
                               basePosition=[0.,0.,0.],
                               useMaximalCoordinates=True)

    return ballId


# Load contact surface in PyBullet for contact experiments
def display_contact_surface(M, robotId=1, radius=.25, length=0.0, bullet_endeff_ids=[], TILT=[0., 0., 0.]):
    '''
    Creates contact surface object in PyBullet as a flat cylinder 
      M              : contact placement expressed in simulator WORLD frame
      robotId        : id of the robot in simulator
      radius         : radius of cylinder
      length         : length of cylinder
      TILT           : RPY tilt of the surface
    '''
    print("Creating PyBullet contact surface...")
    # Tilt contact surface (default 0)
    TILT_rotation = pin.utils.rpyToMatrix(TILT[0], TILT[1], TILT[2])
    M.rotation = TILT_rotation.dot(M.rotation)
    # Get quaternion
    quat = pin.SE3ToXYZQUAT(M)
    visualShapeId = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                                        radius=radius,
                                        length=length,
                                        rgbaColor=[.1, .8, .1, .5],
                                        visualFramePosition=quat[:3],
                                        visualFrameOrientation=quat[3:])
    # With collision
    if(len(bullet_endeff_ids)!=0):
      collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                radius=radius,
                                                height=length,
                                                collisionFramePosition=quat[:3],
                                                collisionFrameOrientation=quat[3:])
      contactId = p.createMultiBody(baseMass=0.,
                                    baseInertialFramePosition=[0.,0.,0.],
                                    baseCollisionShapeIndex=collisionShapeId,
                                    baseVisualShapeIndex=visualShapeId,
                                    basePosition=[0.,0.,0.],
                                    useMaximalCoordinates=False)
                    
      # Desactivate collisions for all links
      for i in range(p.getNumJoints(robotId)):
            p.setCollisionFilterPair(contactId, robotId, -1, i, 1) # 0
            print("Set collision pair ("+str(contactId)+","+str(robotId)+"."+str(i)+") to True")
    #   # activate collisions only for EE ids
    #   for ee_id in bullet_endeff_ids:
    #         p.setCollisionFilterPair(contactId, robotId, -1, ee_id, 1)
    #         logger.info("Set collision pair ("+str(contactId)+","+str(robotId)+"."+str(ee_id)+") to True")
      return contactId
    # Without collisions
    else:
      contactId = p.createMultiBody(baseMass=0.,
                        baseInertialFramePosition=[0.,0.,0.],
                        baseVisualShapeIndex=visualShapeId,
                        basePosition=[0.,0.,0.],
                        useMaximalCoordinates=False)
      return contactId



# Initialize simulation data for MPC simulation
def init_sim_data(sim_params, ocp_params, x0):
    '''
    Initialize simulation data from config file
        sim_params : dict of sim params
        N_h        : number of nodes in OCP horizon
        x0         : initial state of the
    '''
    sim_data = {}
    # MPC & simulation parameters
    sim_data['T_sim'] = sim_params['T_sim']                             # Total duration of simulation (s)
    sim_data['sim_freq'] = sim_params['sim_freq']                       # Simulation frequency
    sim_data['mpc_freq'] = sim_params['mpc_freq']                       # Planning frequency (OCP solution update rate)
    sim_data['N_mpc'] = int(sim_data['T_sim']*sim_data['mpc_freq'])    # Total number of planning steps in the simulation
    sim_data['N_sim'] = int(sim_data['T_sim']*sim_data['sim_freq'])     # Total number of simulation steps 
    sim_data['dt_mpc'] = float(1./sim_data['mpc_freq'])                 # Duration of 1 planning cycle (s)
    sim_data['dt_sim'] = float(1./sim_data['sim_freq'])                # Duration of 1 simulation cycle (s)
    sim_data['ocp_to_mpc_ratio'] = sim_data['dt_mpc']/ocp_params['dt'] 
    # Copy OCP params
    sim_data['nq'] = ocp_params['pin_model'].nq
    sim_data['nv'] = ocp_params['pin_model'].nv
    sim_data['nu'] = ocp_params['pin_model'].nq
    sim_data['nx'] = sim_data['nq'] + sim_data['nv']
    sim_data['pin_model'] = ocp_params['pin_model']
    sim_data['id_endeff'] = ocp_params['id_endeff']
    sim_data['armature']  = ocp_params['armature']
    sim_data['T_h'] = ocp_params['N_h']*ocp_params['dt']                # Duration of the MPC horizon (s)
    sim_data['N_h'] = ocp_params['N_h']                                 # Number of nodes in MPC horizon
    sim_data['active_costs'] = ocp_params['active_costs']               # List of ative costs names
    # Cost references 
    sim_data['ctrl_ref'] = np.zeros((sim_data['N_mpc'], sim_data['nu']))
    sim_data['state_ref'] = np.zeros((sim_data['N_mpc'], sim_data['nx']))
    sim_data['lin_pos_ee_ref'] = np.zeros((sim_data['N_mpc'], 3))
    sim_data['lin_vel_ee_ref'] = np.zeros((sim_data['N_mpc'], 3))
    sim_data['ang_pos_ee_ref'] = np.zeros((sim_data['N_mpc'], 3))
    sim_data['ang_vel_ee_ref'] = np.zeros((sim_data['N_mpc'], 3))
    sim_data['f_ee_ref'] = np.zeros((sim_data['N_mpc'], 6))
    # Predictions
    sim_data['state_pred'] = np.zeros((sim_data['N_mpc'], ocp_params['N_h']+1, sim_data['nx'])) # Predicted states  ( ddp.xs : {x* = (q*, v*)} )
    sim_data['ctrl_pred'] = np.zeros((sim_data['N_mpc'], ocp_params['N_h'], sim_data['nu']))    # Predicted torques ( ddp.us : {u*} )
    sim_data['force_pred'] = np.zeros((sim_data['N_mpc'], ocp_params['N_h'], 6))                # Predicted EE contact forces
    sim_data['state_des_MPC_RATE'] = np.zeros((sim_data['N_mpc']+1, sim_data['nx']))                # Predicted states at planner frequency  ( x* interpolated at PLAN freq )
    sim_data['ctrl_des_MPC_RATE'] = np.zeros((sim_data['N_mpc'], sim_data['nu']))                   # Predicted torques at planner frequency ( u* interpolated at PLAN freq )
    sim_data['force_des_MPC_RATE'] = np.zeros((sim_data['N_mpc'], 6))                               # Predicted EE contact forces planner frequency  
    sim_data['state_des_SIM_RATE'] = np.zeros((sim_data['N_sim']+1, sim_data['nx']))                # Reference state at actuation freq ( x* interpolated at SIMU freq )
    sim_data['ctrl_des_SIM_RATE'] = np.zeros((sim_data['N_sim'], sim_data['nu']))                   # Reference input at actuation freq ( u* interpolated at SIMU freq )
    sim_data['force_des_SIM_RATE'] = np.zeros((sim_data['N_sim'], 6))                               # Reference EE contact force at actuation freq
    # Measurements
    sim_data['state_mea_SIM_RATE'] = np.zeros((sim_data['N_sim']+1, sim_data['nx']))            # Measured states ( x^mea = (q, v) from actuator & PyB at SIMU freq )
    sim_data['force_mea_SIM_RATE'] = np.zeros((sim_data['N_sim'], 6)) 
    sim_data['state_mea_SIM_RATE'][0, :] = x0
    print('')
    print('                       *************************')
    print('                       ** Simulation is ready **') 
    print('                       *************************')        
    print("-------------------------------------------------------------------")
    print('- Total simulation duration            : T_sim           = '+str(sim_data['T_sim'])+' s')
    print('- Simulation frequency                 : f_simu          = '+str(float(sim_data['sim_freq']/1000.))+' kHz')
    print('- Replanning frequency                 : f_plan          = '+str(float(sim_data['mpc_freq']/1000.))+' kHz')
    print('- Total # of simulation steps          : N_sim          = '+str(sim_data['N_sim']))
    print('- Total # of planning steps            : N_mpc          = '+str(sim_data['N_mpc']))
    print('- Duration of MPC horizon              : T_ocp           = '+str(sim_data['T_h'])+' s')
    print('- OCP integration step                 : dt              = '+str(ocp_params['dt'])+' s')
    print("-------------------------------------------------------------------")
    print('')
    # time.sleep(2)

    return sim_data


# Extract MPC simu-specific plotting data from sim data
def extract_plot_data_from_sim_data(sim_data):
    '''
    Extract plot data from simu data
    '''
    
    plot_data = {}
    # Get costs
    plot_data['active_costs'] = sim_data['active_costs']
    # Robot model & params
    plot_data['pin_model'] = sim_data['pin_model']
    nq = plot_data['pin_model'].nq; plot_data['nq'] = nq
    nv = plot_data['pin_model'].nv; plot_data['nv'] = nv
    nx = nq+nv; plot_data['nx'] = nx
    nu = nq
    # MPC params
    plot_data['T_sim']  = sim_data['T_sim']
    plot_data['N_sim']  = sim_data['N_sim']; plot_data['N_mpc'] = sim_data['N_mpc']
    plot_data['dt_mpc'] = sim_data['dt_mpc']; plot_data['dt_sim'] = sim_data['dt_sim']
    plot_data['T_h']    = sim_data['T_h']; 
    plot_data['N_h']    = sim_data['N_h']
    # plot_data['alpha']  = sim_data['alpha']; plot_data['beta'] = sim_data['beta']
    # Record cost references
    plot_data['ctrl_ref']       = sim_data['ctrl_ref']
    plot_data['state_ref']      = sim_data['state_ref']
    plot_data['lin_pos_ee_ref'] = sim_data['lin_pos_ee_ref']
    plot_data['f_ee_ref']       = sim_data['f_ee_ref']
    # Control predictions
    plot_data['u_pred']         = sim_data['ctrl_pred']
    plot_data['u_des_MPC_RATE'] = sim_data['ctrl_des_MPC_RATE']
    plot_data['u_des_SIM_RATE'] = sim_data['ctrl_des_SIM_RATE']
    # State predictions (at PLAN freq)
    plot_data['q_pred']         = sim_data['state_pred'][:,:,:nq]
    plot_data['v_pred']         = sim_data['state_pred'][:,:,nq:nq+nv]
    plot_data['q_des_MPC_RATE'] = sim_data['state_des_MPC_RATE'][:,:nq]
    plot_data['v_des_MPC_RATE'] = sim_data['state_des_MPC_RATE'][:,nq:nq+nv] 
    plot_data['q_des_SIM_RATE'] = sim_data['state_des_SIM_RATE'][:,:nq]
    plot_data['v_des_SIM_RATE'] = sim_data['state_des_SIM_RATE'][:,nq:nq+nv]
    # State measurements (at SIMU freq)
    plot_data['q_mea'] = sim_data['state_mea_SIM_RATE'][:,:nq]
    plot_data['v_mea'] = sim_data['state_mea_SIM_RATE'][:,nq:nq+nv]
    # Extract gravity torques
    plot_data['grav'] = np.zeros((sim_data['N_sim']+1, plot_data['nq']))
    for i in range(plot_data['N_sim']+1):
      plot_data['grav'][i,:] = pin_utils.get_u_grav(plot_data['q_mea'][i,:], plot_data['pin_model'], sim_data['armature'])
    # EE predictions (at PLAN freq)
      # Linear position velocity of EE
    plot_data['lin_pos_ee_pred'] = np.zeros((sim_data['N_mpc'], sim_data['N_h']+1, 3))
    plot_data['lin_vel_ee_pred'] = np.zeros((sim_data['N_mpc'], sim_data['N_h']+1, 3))
      # Angular position velocity of EE
    plot_data['ang_pos_ee_pred'] = np.zeros((sim_data['N_mpc'], sim_data['N_h']+1, 3)) 
    plot_data['ang_vel_ee_pred'] = np.zeros((sim_data['N_mpc'], sim_data['N_h']+1, 3)) 
    for node_id in range(sim_data['N_h']+1):
        plot_data['lin_pos_ee_pred'][:, node_id, :] = pin_utils.get_p_(plot_data['q_pred'][:, node_id, :], plot_data['pin_model'], sim_data['id_endeff'])
        plot_data['lin_vel_ee_pred'][:, node_id, :] = pin_utils.get_v_(plot_data['q_pred'][:, node_id, :], plot_data['v_pred'][:, node_id, :], plot_data['pin_model'], sim_data['id_endeff'])
        plot_data['ang_pos_ee_pred'][:, node_id, :] = pin_utils.get_rpy_(plot_data['q_pred'][:, node_id, :], plot_data['pin_model'], sim_data['id_endeff'])
        plot_data['ang_vel_ee_pred'][:, node_id, :] = pin_utils.get_w_(plot_data['q_pred'][:, node_id, :], plot_data['v_pred'][:, node_id, :], plot_data['pin_model'], sim_data['id_endeff'])
    # EE measurements (at SIMU freq)
      # Linear
    plot_data['lin_pos_ee_mea'] = pin_utils.get_p_(plot_data['q_mea'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_vel_ee_mea'] = pin_utils.get_v_(plot_data['q_mea'], plot_data['v_mea'], sim_data['pin_model'], sim_data['id_endeff'])
      # Angular
    plot_data['ang_pos_ee_mea'] = pin_utils.get_rpy_(plot_data['q_mea'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_vel_ee_mea'] = pin_utils.get_w_(plot_data['q_mea'], plot_data['v_mea'], sim_data['pin_model'], sim_data['id_endeff'])
    # EE des
      # Linear
    plot_data['lin_pos_ee_des_MPC_RATE'] = pin_utils.get_p_(plot_data['q_des_MPC_RATE'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_vel_ee_des_MPC_RATE'] = pin_utils.get_v_(plot_data['q_des_MPC_RATE'], plot_data['v_des_MPC_RATE'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_pos_ee_des_SIM_RATE'] = pin_utils.get_p_(plot_data['q_des_SIM_RATE'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['lin_vel_ee_des_SIM_RATE'] = pin_utils.get_v_(plot_data['q_des_SIM_RATE'], plot_data['v_des_SIM_RATE'], sim_data['pin_model'], sim_data['id_endeff'])
      # Angular
    plot_data['ang_pos_ee_des_MPC_RATE'] = pin_utils.get_rpy_(plot_data['q_des_MPC_RATE'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_vel_ee_des_MPC_RATE'] = pin_utils.get_w_(plot_data['q_des_MPC_RATE'], plot_data['v_des_MPC_RATE'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_pos_ee_des_SIM_RATE'] = pin_utils.get_rpy_(plot_data['q_des_SIM_RATE'], sim_data['pin_model'], sim_data['id_endeff'])
    plot_data['ang_vel_ee_des_SIM_RATE'] = pin_utils.get_w_(plot_data['q_des_SIM_RATE'], plot_data['v_des_SIM_RATE'], sim_data['pin_model'], sim_data['id_endeff'])
    # Extract EE force
    plot_data['f_ee_pred']         = sim_data['force_pred']
    plot_data['f_ee_mea']          = sim_data['force_mea_SIM_RATE']
    plot_data['f_ee_des_MPC_RATE'] = sim_data['force_des_MPC_RATE']
    plot_data['f_ee_des_SIM_RATE'] = sim_data['force_des_SIM_RATE']

    return plot_data


# Plot from MPC simulation
def plot_mpc_results(plot_data, which_plots=None, PLOT_PREDICTIONS=False, 
                                              pred_plot_sampling=100, 
                                              SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                                              SHOW=True,
                                              AUTOSCALE=False):
    '''
    Plot sim data
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
      AUTOSCALE                 : rescale y-axis of endeff plot 
                                  based on maximum value taken
    '''

    plots = {}

    if('x' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        plots['x'] = plot_mpc_state(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                           pred_plot_sampling=pred_plot_sampling, 
                                           SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                           SHOW=False, AUTOSCALE=AUTOSCALE)
    
    if('u' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        plots['u'] = plot_mpc_control(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                             pred_plot_sampling=pred_plot_sampling, 
                                             SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                             SHOW=False, AUTOSCALE=AUTOSCALE)

    if('ee' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        plots['ee_lin'] = plot_mpc_endeff_linear(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                            pred_plot_sampling=pred_plot_sampling, 
                                            SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False, AUTOSCALE=AUTOSCALE)
        plots['ee_ang'] = plot_mpc_endeff_angular(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                            pred_plot_sampling=pred_plot_sampling, 
                                            SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False, AUTOSCALE=AUTOSCALE)

    if('f' in which_plots or which_plots is None or which_plots =='all' or 'all' in which_plots):
        plots['f'] = plot_mpc_force(plot_data, PLOT_PREDICTIONS=PLOT_PREDICTIONS, 
                                            pred_plot_sampling=pred_plot_sampling, 
                                            SAVE=SAVE, SAVE_DIR=SAVE_DIR, SAVE_NAME=SAVE_NAME,
                                            SHOW=False, AUTOSCALE=AUTOSCALE)    
    if(SHOW):
        plt.show() 
    plt.close('all')

# Plot state data
def plot_mpc_state(plot_data, PLOT_PREDICTIONS=False, 
                          pred_plot_sampling=100, 
                          SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                          SHOW=True, AUTOSCALE=True):
    '''
    Plot state data
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
    '''
    T_sim = plot_data['T_sim']
    N_sim = plot_data['N_sim']
    N_mpc = plot_data['N_mpc']
    dt_mpc = plot_data['dt_mpc']
    nq = plot_data['nq']
    nx = plot_data['nx']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu = np.linspace(0, T_sim, N_sim+1)
    t_span_plan = np.linspace(0, T_sim, N_mpc+1)
    fig_x, ax_x = plt.subplots(nq, 2, figsize=(19.2,10.8), sharex='col') 
    # For each joint
    for i in range(nq):

        if(PLOT_PREDICTIONS):

            # Extract state predictions of i^th joint
            q_pred_i = plot_data['q_pred'][:,:,i]
            v_pred_i = plot_data['v_pred'][:,:,i]
            # For each planning step in the trajectory
            for j in range(0, N_mpc, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_mpc
                tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
                tspan_u_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_mpc, N_h)
                # Set up lists of (x,y) points for predicted positions and velocities
                points_q = np.array([tspan_x_pred, q_pred_i[j,:]]).transpose().reshape(-1,1,2)
                points_v = np.array([tspan_x_pred, v_pred_i[j,:]]).transpose().reshape(-1,1,2)
                # Set up lists of segments
                segs_q = np.concatenate([points_q[:-1], points_q[1:]], axis=1)
                segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
                # Make collections segments
                cm = plt.get_cmap('Greys_r') 
                lc_q = LineCollection(segs_q, cmap=cm, zorder=-1)
                lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
                lc_q.set_array(tspan_x_pred)
                lc_v.set_array(tspan_x_pred) 
                # Customize
                lc_q.set_linestyle('-')
                lc_v.set_linestyle('-')
                lc_q.set_linewidth(1)
                lc_v.set_linewidth(1)
                # Plot collections
                ax_x[i,0].add_collection(lc_q)
                ax_x[i,1].add_collection(lc_v)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                my_colors = cm(colors)
                ax_x[i,0].scatter(tspan_x_pred, q_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black', 
                ax_x[i,1].scatter(tspan_x_pred, v_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys) #c='black',

        # Joint position
        ax_x[i,0].plot(t_span_plan, plot_data['q_des_MPC_RATE'][:,i], color='b', linestyle='-', marker='.', label='Predicted', alpha=0.1)
        # ax_x[i,0].plot(t_span_simu, plot_data['q_des_SIM_RATE'][:,i], color='y', linestyle='-', marker='.', label='Predicted (SIMU rate)', alpha=0.5)
        ax_x[i,0].plot(t_span_simu, plot_data['q_mea'][:,i], 'r-', label='Measured', linewidth=1, alpha=0.3)
        # Plot joint position regularization reference
        if('stateReg' in plot_data['active_costs']):
            ax_x[i,0].plot(t_span_plan[:-1], plot_data['state_ref'][:, i], linestyle='-.', color='k', marker=None, label='xReg_ref', alpha=0.5)
        ax_x[i,0].set_ylabel('$q_{}$'.format(i), fontsize=12)
        ax_x[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_x[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax_x[i,0].grid(True)

        # Joint velocity 
        ax_x[i,1].plot(t_span_plan, plot_data['v_des_MPC_RATE'][:,i], color='b', linestyle='-', marker='.', label='Predicted', alpha=0.5)
        # ax_x[i,1].plot(t_span_simu, plot_data['v_des_SIM_RATE'][:,i], color='y', linestyle='-', marker='.', label='Predicted (SIMU)', alpha=0.5)
        ax_x[i,1].plot(t_span_simu, plot_data['v_mea'][:,i], 'r-', label='Measured', linewidth=1, alpha=0.3)
        if('stateReg' in plot_data['active_costs']):
            ax_x[i,1].plot(t_span_plan[:-1], plot_data['state_ref'][:, i+nq], linestyle='-.', color='k', marker=None, label='xReg_ref', alpha=0.5)
        ax_x[i,1].set_ylabel('$v_{}$'.format(i), fontsize=12)
        ax_x[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_x[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax_x[i,1].grid(True)

        # Add xlabel on bottom plot of each column
        if(i == nq-1):
            ax_x[i,0].set_xlabel('t(s)', fontsize=16)
            ax_x[i,1].set_xlabel('t(s)', fontsize=16)
        # Legend
        handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
        fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
    # y axis labels
    fig_x.text(0.05, 0.5, 'Joint position (rad)', va='center', rotation='vertical', fontsize=16)
    fig_x.text(0.49, 0.5, 'Joint velocity (rad/s)', va='center', rotation='vertical', fontsize=16)
    fig_x.subplots_adjust(wspace=0.27)
    # Titles
    fig_x.suptitle('State = joint positions, velocities', size=18)
    if(AUTOSCALE):
        ax_q_lim = np.pi
        ax_v_lim = 2.
        for i in range(nq):
            ax_x[i,0].set_ylim(-ax_q_lim, ax_q_lim) 
            ax_x[i,1].set_ylim(-ax_v_lim, ax_v_lim) 
    # Save fig
    if(SAVE):
        figs = {'x': fig_x}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig_x

# Plot control data
def plot_mpc_control(plot_data, PLOT_PREDICTIONS=False, 
                            pred_plot_sampling=100, 
                            SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                            SHOW=True,
                            AUTOSCALE=False):
    '''
    Plot control data
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
    '''
    T_sim = plot_data['T_sim']
    N_sim = plot_data['N_sim']
    N_mpc = plot_data['N_mpc']
    dt_mpc = plot_data['dt_mpc']
    dt_sim = plot_data['dt_sim']
    nq = plot_data['nq']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu = np.linspace(0, T_sim-dt_sim, N_sim)
    t_span_plan = np.linspace(0, T_sim-dt_mpc, N_mpc)
    fig_u, ax_u = plt.subplots(nq, 1, figsize=(19.2,10.8), sharex='col') 
    # For each joint
    for i in range(nq):

        if(PLOT_PREDICTIONS):

            # Extract state predictions of i^th joint
            u_pred_i = plot_data['u_pred'][:,:,i]

            # For each planning step in the trajectory
            for j in range(0, N_mpc, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_mpc
                tspan_u_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_mpc, N_h)
                # Set up lists of (x,y) points for predicted positions and velocities
                points_u = np.array([tspan_u_pred, u_pred_i[j,:]]).transpose().reshape(-1,1,2)
                # Set up lists of segments
                segs_u = np.concatenate([points_u[:-1], points_u[1:]], axis=1)
                # Make collections segments
                cm = plt.get_cmap('Greys_r') 
                lc_u = LineCollection(segs_u, cmap=cm, zorder=-1)
                lc_u.set_array(tspan_u_pred)
                # Customize
                lc_u.set_linestyle('-')
                lc_u.set_linewidth(1)
                # Plot collections
                ax_u[i].add_collection(lc_u)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                my_colors = cm(colors)
                ax_u[i].scatter(tspan_u_pred, u_pred_i[j,:], s=10, zorder=1, c=cm(np.r_[np.linspace(0.1, 1, N_h-1), 1] ), cmap=matplotlib.cm.Greys) #c='black' 

        # Joint torques
        ax_u[i].plot(t_span_plan, plot_data['u_pred'][:,0,i], color='r', marker=None, linestyle='-', label='Optimal control u0*', alpha=0.6)
        ax_u[i].plot(t_span_plan, plot_data['u_des_MPC_RATE'][:,i], color='b', linestyle='-', marker='.', label='Predicted', alpha=0.1)
        # ax_u[i].plot(t_span_simu, plot_data['u_des_SIM_RATE'][:,i], color='y', linestyle='-', marker='.', label='Prediction (SIMU)', alpha=0.6)
        # ax_u[i].plot(t_span_simu, plot_data['grav'][:-1,i], color=[0.,1.,0.,0.], marker=None, linestyle='-.', label='Gravity torque', alpha=0.9)
        # Plot reference
        if('ctrlReg' or 'ctrlRegGrav' in plot_data['active_costs']):
            ax_u[i].plot(t_span_plan, plot_data['ctrl_ref'][:, i], linestyle='-.', color='k', marker=None, label='uReg_ref', alpha=0.5)
        ax_u[i].set_ylabel('$u_{}$'.format(i), fontsize=12)
        ax_u[i].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_u[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax_u[i].grid(True)
        # Last x axis label
        if(i == nq-1):
            ax_u[i].set_xlabel('t (s)', fontsize=16)
        # LEgend
        handles_u, labels_u = ax_u[i].get_legend_handles_labels()
        fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})
    # Sup-y label
    fig_u.text(0.04, 0.5, 'Joint torque (Nm)', va='center', rotation='vertical', fontsize=16)
    # Titles
    fig_u.suptitle('Control = joint torques', size=18)
    if(AUTOSCALE):
        ax_u_lim = [300, 300, 150, 150, 70, 10, 10]
        for i in range(nq):
            ax_u[i].set_ylim(-ax_u_lim[i], ax_u_lim[i]) 
    # Save figs
    if(SAVE):
        figs = {'u': fig_u}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 

    return fig_u

# Plot end-eff data
def plot_mpc_endeff_linear(plot_data, PLOT_PREDICTIONS=False, 
                               pred_plot_sampling=100, 
                               SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                               SHOW=True,
                               AUTOSCALE=False):
    '''
    Plot endeff data (linear position and velocity)
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
      AUTOSCALE                 : rescale y-axis of endeff plot 
                                  based on maximum value taken
    '''
    T_sim = plot_data['T_sim']
    N_sim = plot_data['N_sim']
    N_mpc = plot_data['N_mpc']
    dt_mpc = plot_data['dt_mpc']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu = np.linspace(0, T_sim, N_sim+1)
    t_span_plan = np.linspace(0, T_sim, N_mpc+1)
    fig, ax = plt.subplots(3, 2, figsize=(19.2,10.8), sharex='col') 
    # Plot endeff
    xyz = ['x', 'y', 'z']
    for i in range(3):

        if(PLOT_PREDICTIONS):
            lin_pos_ee_pred_i = plot_data['lin_pos_ee_pred'][:, :, i]
            lin_vel_ee_pred_i = plot_data['lin_vel_ee_pred'][:, :, i]
            # For each planning step in the trajectory
            for j in range(0, N_mpc, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_mpc
                tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
                # Set up lists of (x,y) points for predicted positions
                points_p = np.array([tspan_x_pred, lin_pos_ee_pred_i[j,:]]).transpose().reshape(-1,1,2)
                points_v = np.array([tspan_x_pred, lin_vel_ee_pred_i[j,:]]).transpose().reshape(-1,1,2)
                # Set up lists of segments
                segs_p = np.concatenate([points_p[:-1], points_p[1:]], axis=1)
                segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
                # Make collections segments
                cm = plt.get_cmap('Greys_r') 
                lc_p = LineCollection(segs_p, cmap=cm, zorder=-1)
                lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
                lc_p.set_array(tspan_x_pred)
                lc_v.set_array(tspan_x_pred)
                # Customize
                lc_p.set_linestyle('-')
                lc_v.set_linestyle('-')
                lc_p.set_linewidth(1)
                lc_v.set_linewidth(1)
                # Plot collections
                ax[i,0].add_collection(lc_p)
                ax[i,1].add_collection(lc_v)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                my_colors = cm(colors)
                ax[i,0].scatter(tspan_x_pred, lin_pos_ee_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)
                ax[i,1].scatter(tspan_x_pred, lin_vel_ee_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)

        # EE position
        ax[i,0].plot(t_span_plan, plot_data['lin_pos_ee_des_MPC_RATE'][:,i], color='b', linestyle='-', marker='.', label='Predicted ', alpha=0.1)
        # ax[i,0].plot(t_span_simu, plot_data['lin_pos_ee_des_SIM_RATE'][:,i], color='y', linestyle='-', marker='.', label='Predicted (SIMU rate)', alpha=0.5)
        ax[i,0].plot(t_span_simu, plot_data['lin_pos_ee_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        # Plot reference
        if('translation' in plot_data['active_costs']):
            ax[i,0].plot(t_span_plan[:-1], plot_data['lin_pos_ee_ref'][:,i], color='k', linestyle='-.', linewidth=2., label='Reference', alpha=0.9)
        ax[i,0].set_ylabel('$P^{EE}_%s$  (m)'%xyz[i], fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax[i,0].grid(True)
        
        # EE velocity
        ax[i,1].plot(t_span_plan, plot_data['lin_vel_ee_des_MPC_RATE'][:,i], color='b', linestyle='-', marker='.', label='Predicted ', alpha=0.1)
        # ax[i,1].plot(t_span_simu, plot_data['lin_vel_ee_des_SIM_RATE'][:,i], color='y', linestyle='-', marker='.', label='Predicted (SIMU rate)', alpha=0.5)
        ax[i,1].plot(t_span_simu, plot_data['lin_vel_ee_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        # Plot reference 
        if('velocity' in plot_data['active_costs']):
            ax[i,1].plot(t_span_plan, [0.]*(N_mpc+1), color='k', linestyle='-.', linewidth=2., label='Reference', alpha=0.9)
        ax[i,1].set_ylabel('$V^{EE}_%s$  (m)'%xyz[i], fontsize=16)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax[i,1].grid(True)


    # Align
    fig.align_ylabels(ax[:,0])
    fig.align_ylabels(ax[:,1])
    ax[i,0].set_xlabel('t (s)', fontsize=16)
    ax[i,1].set_xlabel('t (s)', fontsize=16)
    # Set ylim if any
    TOL = 1e-3
    if(AUTOSCALE):
        ax_p_ylim = 1.1*max(np.max(np.abs(plot_data['lin_pos_ee_mea'])), TOL)
        ax_v_ylim = 1.1*max(np.max(np.abs(plot_data['lin_vel_ee_mea'])), TOL)
        for i in range(3):
            ax[i,0].set_ylim(-ax_p_ylim, ax_p_ylim) 
            ax[i,1].set_ylim(-ax_v_ylim, ax_v_ylim) 

    handles_p, labels_p = ax[0,0].get_legend_handles_labels()
    fig.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})
    # Titles
    fig.suptitle('End-effector trajectories', size=18)
    # Save figs
    if(SAVE):
        figs = {'ee_lin': fig}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig, ax

# Plot end-eff data
def plot_mpc_endeff_angular(plot_data, PLOT_PREDICTIONS=False, 
                               pred_plot_sampling=100, 
                               SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                               SHOW=True,
                               AUTOSCALE=False):
    '''
    Plot endeff data (angular position and velocity)
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
      AUTOSCALE                 : rescale y-axis of endeff plot 
                                  based on maximum value taken
    '''
    T_sim = plot_data['T_sim']
    N_sim = plot_data['N_sim']
    N_mpc = plot_data['N_mpc']
    dt_mpc = plot_data['dt_mpc']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu = np.linspace(0, T_sim, N_sim+1)
    t_span_plan = np.linspace(0, T_sim, N_mpc+1)
    fig, ax = plt.subplots(3, 2, figsize=(19.2,10.8), sharex='col') 
    # Plot endeff
    xyz = ['x', 'y', 'z']
    for i in range(3):

        if(PLOT_PREDICTIONS):
            ang_pos_ee_pred_i = plot_data['ang_pos_ee_pred'][:, :, i]
            ang_vel_ee_pred_i = plot_data['ang_vel_ee_pred'][:, :, i]
            # For each planning step in the trajectory
            for j in range(0, N_mpc, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_mpc
                tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h, N_h+1)
                # Set up lists of (x,y) points for predicted positions
                points_p = np.array([tspan_x_pred, ang_pos_ee_pred_i[j,:]]).transpose().reshape(-1,1,2)
                points_v = np.array([tspan_x_pred, ang_vel_ee_pred_i[j,:]]).transpose().reshape(-1,1,2)
                # Set up lists of segments
                segs_p = np.concatenate([points_p[:-1], points_p[1:]], axis=1)
                segs_v = np.concatenate([points_v[:-1], points_v[1:]], axis=1)
                # Make collections segments
                cm = plt.get_cmap('Greys_r') 
                lc_p = LineCollection(segs_p, cmap=cm, zorder=-1)
                lc_v = LineCollection(segs_v, cmap=cm, zorder=-1)
                lc_p.set_array(tspan_x_pred)
                lc_v.set_array(tspan_x_pred)
                # Customize
                lc_p.set_linestyle('-')
                lc_v.set_linestyle('-')
                lc_p.set_linewidth(1)
                lc_v.set_linewidth(1)
                # Plot collections
                ax[i,0].add_collection(lc_p)
                ax[i,1].add_collection(lc_v)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h), 1] 
                my_colors = cm(colors)
                ax[i,0].scatter(tspan_x_pred, ang_pos_ee_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)
                ax[i,1].scatter(tspan_x_pred, ang_vel_ee_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)

        # EE position
        ax[i,0].plot(t_span_plan, plot_data['ang_pos_ee_des_MPC_RATE'][:,i], color='b', linestyle='-', marker='.', label='Predicted ', alpha=0.1)
        # ax[i,0].plot(t_span_simu, plot_data['ang_pos_ee_des_SIM_RATE'][:,i], color='y', linestyle='-', marker='.', label='Predicted (SIMU rate)', alpha=0.5)
        ax[i,0].plot(t_span_simu, plot_data['ang_pos_ee_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        # Plot reference
        if('rotation' in plot_data['active_costs']):
            ax[i,0].plot(t_span_plan[:-1], plot_data['ang_pos_ee_ref'][:,i], 'm-.', linewidth=2., label='Reference', alpha=0.9)
        ax[i,0].set_ylabel('$RPY^{EE}_%s$  (m)'%xyz[i], fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax[i,0].grid(True)
        
        # EE velocity
        ax[i,1].plot(t_span_plan, plot_data['ang_vel_ee_des_MPC_RATE'][:,i], color='b', linestyle='-', marker='.', label='Predicted ', alpha=0.1)
        # ax[i,1].plot(t_span_simu, plot_data['ang_vel_ee_des_SIM_RATE'][:,i], color='y', linestyle='-', marker='.', label='Predicted (SIMU rate)', alpha=0.5)
        ax[i,1].plot(t_span_simu, plot_data['ang_vel_ee_mea'][:,i], 'r-', label='Measured (WITH noise)', linewidth=1, alpha=0.3)
        # Plot reference 
        if('velocity' in plot_data['active_costs']):
            ax[i,1].plot(t_span_plan, [0.]*(N_mpc+1), 'm-.', linewidth=2., label='Reference', alpha=0.9)
        ax[i,1].set_ylabel('$W^{EE}_%s$  (m)'%xyz[i], fontsize=16)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax[i,1].grid(True)


    # Align
    fig.align_ylabels(ax[:,0])
    fig.align_ylabels(ax[:,1])
    ax[i,0].set_xlabel('t (s)', fontsize=16)
    ax[i,1].set_xlabel('t (s)', fontsize=16)
    # Set ylim if any
    TOL = 1e-3
    if(AUTOSCALE):
        ax_p_ylim = 1.1*max(np.max(np.abs(plot_data['ang_pos_ee_mea'])), TOL)
        ax_v_ylim = 1.1*max(np.max(np.abs(plot_data['ang_vel_ee_mea'])), TOL)
        for i in range(3):
            ax[i,0].set_ylim(-ax_p_ylim, ax_p_ylim) 
            ax[i,1].set_ylim(-ax_v_ylim, ax_v_ylim) 

    handles_p, labels_p = ax[0,0].get_legend_handles_labels()
    fig.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})
    # Titles
    fig.suptitle('End-effector frame orientation (RPY) and angular velocity', size=18)
    # Save figs
    if(SAVE):
        figs = {'ee_ang': fig}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig, ax

# Plot end-eff data
def plot_mpc_force(plot_data, PLOT_PREDICTIONS=False, 
                           pred_plot_sampling=100, 
                           SAVE=False, SAVE_DIR=None, SAVE_NAME=None,
                           SHOW=True,
                           AUTOSCALE=False):
    '''
    Plot EE force data
     Input:
      plot_data                 : plotting data
      PLOT_PREDICTIONS          : True or False
      pred_plot_sampling        : plot every pred_plot_sampling prediction 
                                  to avoid huge amount of plotted data 
                                  ("1" = plot all)
      SAVE, SAVE_DIR, SAVE_NAME : save plots as .png
      SHOW                      : show plots
      AUTOSCALE                 : rescale y-axis of endeff plot 
                                  based on maximum value taken
    '''
    T_sim = plot_data['T_sim']
    N_sim = plot_data['N_sim']
    N_mpc = plot_data['N_mpc']
    dt_mpc = plot_data['dt_mpc']
    dt_sim = plot_data['dt_sim']
    T_h = plot_data['T_h']
    N_h = plot_data['N_h']
    # Create time spans for X and U + Create figs and subplots
    t_span_simu = np.linspace(0, T_sim - dt_sim, N_sim)
    t_span_plan = np.linspace(0, T_sim - dt_mpc, N_mpc)
    fig, ax = plt.subplots(3, 2, figsize=(19.2,10.8), sharex='col') 
    # Plot endeff
    xyz = ['x', 'y', 'z']
    for i in range(3):

        if(PLOT_PREDICTIONS):
            f_ee_pred_i = plot_data['f_ee_pred'][:, :, i]
            # For each planning step in the trajectory
            for j in range(0, N_mpc, pred_plot_sampling):
                # Receding horizon = [j,j+N_h]
                t0_horizon = j*dt_mpc
                tspan_x_pred = np.linspace(t0_horizon, t0_horizon + T_h - dt_mpc, N_h)
                # Set up lists of (x,y) points for predicted positions
                points_f = np.array([tspan_x_pred, f_ee_pred_i[j,:]]).transpose().reshape(-1,1,2)
                # Set up lists of segments
                segs_f = np.concatenate([points_f[:-1], points_f[1:]], axis=1)
                # Make collections segments
                cm = plt.get_cmap('Greys_r') 
                lc_f = LineCollection(segs_f, cmap=cm, zorder=-1)
                lc_f.set_array(tspan_x_pred)
                # Customize
                lc_f.set_linestyle('-')
                lc_f.set_linewidth(1)
                # Plot collections
                ax[i,0].add_collection(lc_f)
                # Scatter to highlight points
                colors = np.r_[np.linspace(0.1, 1, N_h-1), 1] 
                my_colors = cm(colors)
                ax[i,0].scatter(tspan_x_pred, f_ee_pred_i[j,:], s=10, zorder=1, c=my_colors, cmap=matplotlib.cm.Greys)
       
        # EE linear force
        ax[i,0].plot(t_span_plan, plot_data['f_ee_des_MPC_RATE'][:,i], color='b', linestyle='-', marker='.', label='Predicted ', alpha=0.1)
        # ax[i,0].plot(t_span_simu, plot_data['f_ee_des_SIM_RATE'][:,i], color='y', linestyle='-', marker='.', label='Predicted (SIMU rate)', alpha=0.5)
        ax[i,0].plot(t_span_simu, plot_data['f_ee_mea'][:,i], 'r-', label='Measured', linewidth=2, alpha=0.6)
        # Plot reference
        if('force' in plot_data['active_costs']):
            ax[i,0].plot(t_span_plan, plot_data['f_ee_ref'][:,i], color=[0.,1.,0.,0.], linestyle='-.', linewidth=2., label='Reference', alpha=0.9)
        ax[i,0].set_ylabel('$\\lambda^{EE}_%s$  (N)'%xyz[i], fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax[i,0].grid(True)

        # EE angular force 
        ax[i,1].plot(t_span_plan, plot_data['f_ee_des_MPC_RATE'][:,3+i], color='b', linestyle='-', marker='.', label='Predicted ', alpha=0.1)
        # ax[i,1].plot(t_span_simu, plot_data['f_ee_des_SIM_RATE'][:,3+i], color='y', linestyle='-', marker='.', label='Predicted (SIMU rate)', alpha=0.5)
        ax[i,1].plot(t_span_simu, plot_data['f_ee_mea'][:,3+i], 'r-', label='Measured', linewidth=2, alpha=0.6)
        # Plot reference
        if('force' in plot_data['active_costs']):
            ax[i,1].plot(t_span_plan, plot_data['f_ee_ref'][:,3+i], color=[0.,1.,0.,0.], linestyle='-.', linewidth=2., label='Reference', alpha=0.9)
        ax[i,1].set_ylabel('$\\tau^{EE}_%s$  (Nm)'%xyz[i], fontsize=16)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.3e'))
        ax[i,1].grid(True)
    
    # Align
    fig.align_ylabels(ax[:,0])
    fig.align_ylabels(ax[:,1])
    ax[i,0].set_xlabel('t (s)', fontsize=16)
    ax[i,1].set_xlabel('t (s)', fontsize=16)
    # Set ylim if any
    TOL = 1e-3
    if(AUTOSCALE):
        ax_ylim = 1.1*max( np.max(np.abs(plot_data['f_ee_mea'])), np.max(np.abs(plot_data['f_ee_ref'])), TOL )
        ax_ylim = 1.1*max( np.max(np.abs(plot_data['f_ee_mea'])), np.max(np.abs(plot_data['f_ee_ref'])), TOL )
        for i in range(3):
            ax[i,0].set_ylim(-ax_ylim, ax_ylim) 
            # ax[i,0].set_ylim(-30, 10) 
            ax[i,1].set_ylim(-ax_ylim, ax_ylim) 

    handles_p, labels_p = ax[0,0].get_legend_handles_labels()
    fig.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})
    # Titles
    fig.suptitle('End-effector forces', size=18)
    # Save figs
    if(SAVE):
        figs = {'f': fig}
        if(SAVE_DIR is None):
            SAVE_DIR = '/home/skleff/force-feedback/data'
        if(SAVE_NAME is None):
            SAVE_NAME = 'testfig'
        for name, fig in figs.items():
            fig.savefig(SAVE_DIR + '/' +str(name) + '_' + SAVE_NAME +'.png')
    
    if(SHOW):
        plt.show() 
    
    return fig, ax



import numpy as np

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


class AntiAliasingFilter:
  def __init__(self, filter_type='iir'):
      supported_filter_types = ["iir", "moving_average"]
      try: assert(filter_type in supported_filter_types)
      except: logger.error("AntiAliasingFilter type should be in "+str(supported_filter_types))
      self.filter_type = filter_type
      logger.info("Created AntiAliasingFilter of type "+str(self.filter_type))

  def iir(self, last_output, input_data, fc=100, fs=1000):
      '''
       Infinite Impulse Response (IIR) filter 
        output = g*last_output + (1-g)*input_data
        cutoff frequency say 1kHz
        exp(- 2 pi f 1/5000)
      '''
      gamma = np.exp(-2*np.pi*fc/fs)
      return gamma*last_output + (1-gamma)*input_data
      
     
  def step(self, nb1, nb2, freq1, freq2, data_at_freq2):
      '''
      nb1   : number of cycles elapsed at freq1
      nb2   : number of cycles elpased at freq2
      freq1 : low frequency
      freq2 : high frequency 
      '''
      # Using moving average

      try: 
          assert(freq1 <= freq2)
      except:
          logger.error("freq1 must be <= freq2 !!!")
      filterSize = min(nb1, int(freq2/freq1))
      if(filterSize == 0):
          return data_at_freq2[0]
      else:
        data_at_freq1 = self.moving_average(data_at_freq2[nb2-filterSize:nb2], filterSize)
        return data_at_freq1[-1]     
      
  def moving_average(self, input_data, filter_size=1):
      '''
      moving average on 1st dimension of some array ((N,n))
      '''
      output_data = input_data.copy() 
      # Filter them with moving average
      for i in range( output_data.shape[0] ):
          n_sum = min(i, filter_size)
          # Sum up over window
          for k in range(n_sum):
              output_data[i,:] += input_data[i-k-1, :]
          # Divide by number of samples
          output_data[i,:] = output_data[i,:] / (n_sum + 1)
      return output_data 




class LowLevelTorqueController:
    def __init__(self, config, nu, use):
      '''
      Takes in a reference torque (e.g. from MPC) and computes the motor torque 
      to be sent to the robot's motors, optionally using a PID+ controller
      '''
      self.config = config
      self.nu = nu
      # Simulate low-level torque control 
      self.TORQUE_TRACKING   = use           
      logger.info("Created LowLevelTorqueController(TORQUE_TRACKING="+str(self.TORQUE_TRACKING)+").")
      # PID gains for inner control loop 
      self.gain_P = self.config['Kp_low']*np.eye(nu)      
      self.gain_I = self.config['Ki_low']*np.eye(nu)
      self.gain_D = self.config['Kd_low']*np.eye(nu)
      self.err_I  = np.zeros(nu)

    def reset_integral_error(self):
      '''
      Reset integral error to 0
      '''
      self.err_I = np.zeros(self.nu)

    def step(self, reference_torque, measured_torque, measured_torque_derivative):
      '''
      Computes the motor torque
       Input: 
        reference_torque           : desired torque computed by e.g. the MPC 
        measured_torque            : torque measured after actuation effects
        measured_torque_derivative : derivative of the measured torque
       Output:
        motor_torque : torque to be sent to the robot's motor
      '''
      # Feedforward 
      motor_torque = reference_torque.copy()
      # Optional PID feedback term 
      if(self.TORQUE_TRACKING and len(measured_torque) !=0):
          # print(self.TORQUE_TRACKING)
          self.err_P = measured_torque - reference_torque             
          self.err_I += self.err_P
          self.err_D = measured_torque_derivative                 
          motor_torque -= 1.*self.gain_P.dot(self.err_P) 
          motor_torque -= 0.2*self.gain_I.dot(self.err_I)
          motor_torque -= 0.0001*self.gain_D.dot(self.err_D)
      return motor_torque



class ActuationModel:

    def __init__(self, config, nu, SEED=1):
        '''
        Actuation model with parameters defined in config YAML file
        
        The actuation model takes in a motor torque as control input and 
        it computes the combined effects of transmission, noise, friction,
        uncertainty etc. and it returns as an output the measured joint torque.

        It can simulates, depending on the sim parameters 
         - delay
         - noise
         - affine bias
         - dry friction
        '''
        np.random.seed(SEED)
        self.config = config
        self.nu = nu
        # Scaling of desired torque
        self.alpha = np.random.uniform(low=self.config['alpha_min'], high=self.config['alpha_max'], size=(nu,))
        self.beta  = np.random.uniform(low=self.config['beta_min'], high=self.config['beta_max'], size=(nu,))
        # Delays 
        self.delay_sim_cycle = int(self.config['delay_sim_cycle'])       # in simu cycles
        self.buffer_sim      = []                                        # buffer for measured torque delayed by e.g. actuation and/or sensing 
        # Noise
        self.var_u = np.asarray(self.config['var_u'])
        # Actuation model options
        self.DELAY_SIM         = config['DELAY_SIM']                            # Add delay in reference torques (low-level)
        self.SCALE_TORQUES     = config['SCALE_TORQUES']                        # Affinescaling of reference torque
        self.NOISE_TORQUES     = config['NOISE_TORQUES']                        # Moving average smoothing of reference torques
        self.STATIC_FRICTION   = config['STATIC_FRICTION']                      # Simulate static friction
        self.VISCOUS_FRICTION  = config['VISCOUS_FRICTION']                     # Simulate viscous friction
        self.tau_sf_max        = config['static_friction_max_torque']           # Max. static friction torque
        self.tau_vf_slope      = config['viscous_friction_slope']               # Slope of the viscous friction torque w.r.t. velocity
        logger.info("Created ActuationModel(DELAY_SIM="+str(self.DELAY_SIM)+
                    ", SCALE_TORQUES="+str(self.SCALE_TORQUES)+
                    ", NOISE_TORQUES="+str(self.NOISE_TORQUES)+
                    ", STATIC_FRICTION="+str(self.STATIC_FRICTION)+
                    ", VISCOUS_FRICTION="+str(self.VISCOUS_FRICTION)+").")
        if(self.SCALE_TORQUES):
          logger.info("Torques scaling : alpha = "+str(self.alpha)+" | beta = "+str(self.beta))

    def step(self, motor_torque, joint_vel=None):
        '''
        Transforms motor torque into a measured torque
        
        See the paper https://la.disneyresearch.com/wp-content/uploads/Toward-Controlling-a-KUKA-LBR-IIWA-for-Interactive-Tracking-Paper.pdf
          1. FRI receives desired torque tau_d (at 1kHz)
          2. Motor board computes motor torque tau_m = tau_d + PID(tau_d, tau_j)
          tau_j reflects the combined effects of motor inertia, friction, transmission etc. --> approximated by delay, scaling, noise and filtering

        Simulates (optionally) 
         - delay tau_mot(i) = tau_mot(i-delay) due to e.g. transmission
         - affine bias on torques a*tau_mot(i) + b due to model uncertainty
         - Gaussian noise on tau_mot(i) due to torque sensor noise
         - static friction on tau_mot(i) += tanh(a*v)
        Input: 
          i            : current simulation cycle number
          motor_torque : desired torque by motor
          joint_vel    : joint velocity (used for static / viscous friction)
        '''
        # Perfect actuation if all options = False
        measured_torque = motor_torque#.copy()
        # Delay of ref  
        if(self.DELAY_SIM and len(measured_torque) !=0):
          self.buffer_sim.append(measured_torque)            
          if(len(self.buffer_sim) < self.delay_sim_cycle):    
            pass
          else:                          
            measured_torque = self.buffer_sim.pop(-self.delay_sim_cycle)
        # Affine scaling of ref
        if(self.SCALE_TORQUES and len(measured_torque) !=0):
          measured_torque = self.alpha * measured_torque + self.beta
        # Gaussian noise on ref  
        if(self.NOISE_TORQUES and len(measured_torque) !=0):
            noise_u = np.random.normal(0., self.var_u, self.nu)
            measured_torque += noise_u
        #  Static friction
        if(self.STATIC_FRICTION and len(measured_torque) !=0):
           sf_torque = self.tau_sf_max*np.tanh(10*joint_vel) 
          #  print("static friction = ", sf_torque)
           measured_torque -= sf_torque
        if(self.VISCOUS_FRICTION and len(measured_torque) !=0):
           vf_torque = self.tau_vf_slope * joint_vel
          #  print("viscous friction = ", vf_torque)
           measured_torque -= vf_torque
        return measured_torque



class CommunicationModel:

    def __init__(self, config):
        '''
        Communication model with parameters defined in config YAML file
        Simulates (optionally)
         - delay in OCP solution (x*,u*)
        '''
        self.config = config
        # Delay OCP computation
        self.x_buffer_OCP = []                                           # buffer for desired states delayed by OCP computation time
        self.u_buffer_OCP = []                                           # buffer for desired controls delayed by OCP computation time
        # Sensing model options
        self.DELAY_OCP         = config['DELAY_OCP']                     # Add delay in OCP solution (i.e. ~1ms resolution time)
        logger.info("Created CommunicationModel(DELAY_OCP="+str(self.DELAY_OCP)+").")

    def step(self, predicted_state, current_control):
        '''
        Delays input predicted state and current control by 
        using a buffer. Returns the delayed input variables
        Simulates (optionally)
         - delay in OCP solution (x*,u*)
        '''
        # Delay OCP solution due to computation time
        if(self.DELAY_OCP):
          delay = int(self.config['delay_OCP_ms'] * 1e-3 * self.config['plan_freq']) # in planning cycles
          self.x_buffer_OCP.append(predicted_state)
          self.u_buffer_OCP.append(current_control)
          if(len(self.x_buffer_OCP) < delay): 
            pass
          else:                            
            predicted_state = self.x_buffer_OCP.pop(-delay)
          if(len(self.u_buffer_OCP) < delay): 
            pass
          else:
            current_control = self.u_buffer_OCP.pop(-delay)
        return predicted_state, current_control



class SensorModel:

    def __init__(self, config, naug=0, SEED=1):
        '''
        Sensing model with parameters defined in config YAML file
        Simulates (optionally)
         - gaussian noise on measured state
         naug : for augmented state (lpf, soft contact, etc..). 0 by default
        '''
        np.random.seed(SEED)
        self.config = config
        self.nq = len(config['q0']) 
        self.nv = len(config['dq0'])
        self.naug = naug
        # White noise on desired torque and measured state
        self.var_q = np.asarray(self.config['var_q'])
        self.var_v = np.asarray(self.config['var_v'])
        if(self.naug > 0):
          self.var_aug = np.asarray(self.config['var_aug'])[:naug]
        # Sensing model options
        self.NOISE_STATE       = config['NOISE_STATE']                   # Add Gaussian noise on the measured state 
        logger.info("Created SensorModel(NOISE_STATE="+str(self.NOISE_STATE)+").")

    def step(self, measured_state):
        '''
        Transforms simulator state into a measured state
        Simulates (optionally)
         - gaussian noise on measured state
        '''
        # Optional Gaussian noise on measured state 
        if(self.NOISE_STATE):
          noise_q = np.random.normal(0., self.var_q, self.nq)
          noise_v = np.random.normal(0., self.var_v, self.nv)
          if(self.naug > 0):
            noise_tau = np.random.normal(0., self.var_aug, self.naug)
            measured_state += np.concatenate([noise_q, noise_v, noise_tau]).T
          else:
            measured_state += np.concatenate([noise_q, noise_v]).T
        return measured_state


class VelocityEstimator:
  def __init__(self, dt): #, order=1):
    self.dt = dt
    self.q_prev = None
    # self.q_prev2 
    # self.order = order
    # self.nv = len(self.q_prev)
  
  def FD1_estimate(self, q):
    if(self.q_prev is None):
      v = np.zeros(len(q))
    else:
      v = (q - self.q_prev)/self.dt
    self.q_prev = q
    return v

    # else:
  #   return (q - self.q_prev)/self.dt

  # def FD2_estimate(self, q, dt=self.dt):
  #   return q - q_prev
