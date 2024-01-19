from mim_robots.robot_loader import load_pinocchio_wrapper

def get_controlled_joint_ids(robot_name, locked_joints):
    '''
    Returns the ids (in configuration vector q) of the controlled joints
    for a given robot full model and locked joints 
    '''
    pin_robot  = load_pinocchio_wrapper(robot_name)
    full_model = pin_robot.model
    controlled_joint_names = [jn for jn in(full_model.names[1:]) if jn not in locked_joints]
    controlled_joint_ids = [full_model.joints[full_model.getJointId(joint_name)].idx_q for joint_name in controlled_joint_names] 
    return controlled_joint_ids