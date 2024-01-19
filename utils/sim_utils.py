import numpy as np
import pinocchio as pin

import croco_mpc_utils.pinocchio_utils as pin_utils
from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

import pybullet as p 



# Display
def display_ball(p_des, robot_base_pose=pin.SE3.Identity(), RADIUS=.05, COLOR=[1.,1.,1.,1.]):
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
    # logger.debug&("Creating PyBullet sphere visual...")
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
                               useMaximalCoordinates=False)

    return ballId


# Load contact surface in PyBullet for contact experiments
def display_contact_surface(M, robotId=1, radius=.5, length=0.0, bullet_endeff_ids=[], TILT=[0., 0., 0.]):
    '''
    Creates contact surface object in PyBullet as a flat cylinder 
      M              : contact placement expressed in simulator WORLD frame
      robotId        : id of the robot in simulator
      radius         : radius of cylinder
      length         : length of cylinder
      TILT           : RPY tilt of the surface
    '''
    logger.info("Creating PyBullet contact surface...")
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
                    
      # activate collisions for all links
      for i in range(p.getNumJoints(robotId)):
            p.setCollisionFilterPair(contactId, robotId, -1, i, 1) 
      return contactId
    # Without collisions
    else:
      contactId = p.createMultiBody(baseMass=0.,
                        baseInertialFramePosition=[0.,0.,0.],
                        baseVisualShapeIndex=visualShapeId,
                        basePosition=[0.,0.,0.],
                        useMaximalCoordinates=False)
      return contactId


# Load contact surface in PyBullet for contact experiments
def remove_body_from_sim(bodyId):
    '''
    Removes bodyfrom sim env
    '''
    logger.info("Removing body "+str(bodyId)+" from simulation !")
    p.removeBody(bodyId)


def print_dynamics_info(bodyId, linkId=-1):
    '''
    Returns pybullet dynamics info
    '''
    logger.info("Body n°"+str(bodyId))
    d = p.getDynamicsInfo(bodyId, linkId)
    print(d)
    logger.info("  mass                   : "+str(d[0]))
    logger.info("  lateral_friction       : "+str(d[1]))
    logger.info("  local_inertia_diagonal : "+str(d[2]))
    logger.info("  local_inertia_pos      : "+str(d[3]))
    logger.info("  local_inertia_orn      : "+str(d[4]))
    logger.info("  restitution            : "+str(d[5]))
    logger.info("  rolling friction       : "+str(d[6]))
    logger.info("  spinning friction      : "+str(d[7]))
    logger.info("  contact damping        : "+str(d[8]))
    logger.info("  contact stiffness      : "+str(d[9]))
    logger.info("  body type              : "+str(d[10]))
    logger.info("  collision margin       : "+str(d[11]))


# Set lateral friction coefficient to PyBullet body
def set_lateral_friction(bodyId, coef, linkId=-1):
  '''
  Set lateral friction coefficient to PyBullet body
  Input :
    bodyId : PyBullet body unique id
    linkId : linkId . Default : -1 (base link)
    coef   : friction coefficient in (0,1)
  '''
  p.changeDynamics(bodyId, linkId, lateralFriction=coef, rollingFriction=0., spinningFriction=0.) 
  logger.info("Set friction of body n°"+str(bodyId)+" (link n°"+str(linkId)+") to "+str(coef)) 


# Set contact stiffness coefficient to PyBullet body
def set_contact_stiffness_and_damping(bodyId, Ks, Kd, linkId=-1):
  '''
  Set contact stiffness coefficient to PyBullet body
  Input :
    bodyId : PyBullet body unique id
    linkId : linkId . Default : -1 (base link)
    Ks, Kd : stiffness and damping coefficients
  '''
  p.changeDynamics(bodyId, linkId, contactStiffness=Ks, contactDamping=Kd) 
  logger.info("Set contact stiffness of body n°"+str(bodyId)+" (link n°"+str(linkId)+") to "+str(Ks)) 
  logger.info("Set contact damping of body n°"+str(bodyId)+" (link n°"+str(linkId)+") to "+str(Kd)) 


# Set contact stiffness coefficient to PyBullet body
def set_contact_restitution(bodyId, Ks, Kd, linkId=-1):
  '''
  Set contact restitution coefficient to PyBullet body
  Input :
    bodyId : PyBullet body unique id
    linkId : linkId . Default : -1 (base link)
    coef   : restitution coefficient
  '''
  p.changeDynamics(bodyId, linkId, restitution=0.2) 
  logger.info("Set restitution of body n°"+str(bodyId)+" (link n°"+str(linkId)+") to "+str(Ks)) 
