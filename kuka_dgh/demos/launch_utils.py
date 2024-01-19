import yaml
import os
import sys
sys.path.append('.')
from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


# # # # # # # # # # # # # # #
# EXPERIMENT LOADING UTILS  #
# # # # # # # # # # # # # # #

SUPPORTED_EXPERIMENTS = ['polishing',
                         'normal_force']


def is_valid_exp_name(EXP_NAME):
    '''
    Check that exp name is valid
    '''
    try: 
        assert(EXP_NAME in SUPPORTED_EXPERIMENTS)
    except NameError:
        logger.error("Error : config file name must be in "+str(SUPPORTED_EXPERIMENTS))
        
        
def load_config_file(EXP_NAME, path_prefix=''):
    '''
    Load YAML config file corresponding to an experiment name
    '''
    is_valid_exp_name(EXP_NAME)
    config_path = os.path.join(path_prefix, 'config/'+EXP_NAME+".yml")
    logger.debug("Opening config file "+str(config_path))
    with open(config_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def import_mpc_controller(EXP_NAME):
    '''
    Imports the MPC controller class corresponding to an experiment name
    '''
    is_valid_exp_name(EXP_NAME)
    if(EXP_NAME == 'polishing'):     
        from controllers.polishing   import KukaPolishingMPC   as MPCController
    elif(EXP_NAME == 'normal_force'):  
        from controllers.normal_force  import KukaNormalForceMPC  as MPCController
    logger.debug("Imported MPC controller for experiment : "+str(EXP_NAME))
    return MPCController


def get_log_config(EXP_NAME):
    '''
    Returns the log configuration for an experiment name
    '''
    is_valid_exp_name(EXP_NAME)
    if(EXP_NAME == 'polishing'):     
        log_config = POLISHING_LOGS
    elif(EXP_NAME == 'norma_force'):  
        log_config = NORMAL_FORCE_LOGS
    logger.debug("Data log fields : "+str(log_config))
    return log_config



# # # # # # # # # # # # 
# DATA LOGGING UTILS  #
# # # # # # # # # # # # 

LOGS_NONE = []

SSQP_LOGS_MINIMAL = ['KKT', 
                     'ddp_iter',
                     't_child']

POLISHING_LOGS = ['KKT', 
                   'ddp_iter',
                   't_child',
                   'joint_positions',
                   'x_des',
                   'tau',
                   'tau_ff',
                   'tau_gravity',
                   'cost',
                   'joint_torques_measured',
                   'joint_cmd_torques',
                   'target_position']

NORMAL_FORCE_LOGS = ['KKT', 
                    'ddp_iter',
                    't_child',
                    'joint_positions',
                    'joint_velocities',
                    'x_des',
                    'tau',
                    'tau_ff',
                    'tau_gravity',
                    'cost',
                    'joint_torques_measured',
                    'joint_cmd_torques',
                    'target_position_x',
                    'target_position_y',
                    'target_position_z']