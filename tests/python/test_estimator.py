import pinocchio as pin
import example_robot_data as robex
import numpy as np
from numpy.linalg import norm
np.random.seed(10)
from demos.estimator import Estimator, MHEstimator

np.set_printoptions(precision=4, linewidth=180)


pinRefFrame = 'LOCAL_WORLD_ALIGNED' #pin.LOCAL_WORLD_ALIGNED
gains = np.zeros(2)
nc = 1

TOL = 1e-3


# Load robot : works with talos arm, not with kinova
robot_name = 'talos_arm'# 'kinova'  #'talos_arm'
contactFrameName = 'wrist_left_ft_tool_link'  #'j2s6s200_joint_finger_tip_1' # 'wrist_left_ft_tool_link'
robot = robex.load(robot_name)
robot.data = robot.model.createData()
# Initial conditions
q = pin.randomConfiguration(robot.model) 
v = np.random.rand(robot.model.nv) 
a = np.random.rand(robot.model.nv) 
tau = np.random.rand(robot.model.nv) 
f = np.random.rand(nc)
df = np.random.rand(nc)
nq = robot.model.nq
nv = robot.model.nv
id_endeff = robot.model.getFrameId(contactFrameName)

# Create estimators
force_estimator = Estimator(robot, nc, nc, id_endeff, gains, pinRefFrame)
T_MHE = 1
force_estimator_mh = MHEstimator(T_MHE, robot, nc, id_endeff, gains, pinRefFrame)

# Test that they give the same results
_, delta_f = force_estimator.estimate(q, v, a, tau, df, f)
_, delta_f_mh = force_estimator_mh.estimate([q], [v], [a], [tau], [df], [f])

assert(norm(delta_f - delta_f_mh) <= TOL)

print("OK !")