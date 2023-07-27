import pinocchio as pin
import example_robot_data as robex
import numpy as np
from numpy.linalg import norm
np.random.seed(10)
from demos.estimator import Estimator, MHEstimator

np.set_printoptions(precision=4, linewidth=180)


pinRefFrame = 'LOCAL_WORLD_ALIGNED' #pin.LOCAL_WORLD_ALIGNED
pinRefFrame2 = pin.LOCAL_WORLD_ALIGNED
gains = np.zeros(2)
nc = 1

TOL = 1e-3


# Load robot : works with talos arm, not with kinova
robot_name = 'talos_arm'# 'kinova'  #'talos_arm'
contactFrameName = 'wrist_left_ft_tool_link'  #'j2s6s200_joint_finger_tip_1' # 'wrist_left_ft_tool_link'
robot = robex.load(robot_name)
# Initial conditions
q = pin.randomConfiguration(robot.model) 
# Test C++ vs python fails when high velocities : different alpha0 and nu python vs C++ , maybe numpy numerical precision is the issue?
v = np.random.rand(robot.model.nv)*0.01 
# Careful ! In python estimator, zero acc is passed to forward kinematics (not the case in C++ estimator)
a = np.zeros(robot.model.nv) 
tau = np.random.rand(robot.model.nv) 
f = np.random.rand(nc)
df = np.zeros(nc)
nq = robot.model.nq
nv = robot.model.nv
id_endeff = robot.model.getFrameId(contactFrameName)

# Create estimators
force_estimator = Estimator(robot, nc, nc, id_endeff, gains, pinRefFrame)
T_MHE = 1
force_estimator_mh = MHEstimator(T_MHE, robot, nc, id_endeff, gains, pinRefFrame)

# Test that they give the same results
_, delta_f = force_estimator.estimate(q.copy(), v.copy(), a.copy(), tau.copy(), df.copy(), f.copy())
_, delta_f_mh = force_estimator_mh.estimate([q.copy()], [v.copy()], [a.copy()], [tau.copy()], [df.copy()], [f.copy()])
assert(norm(delta_f - delta_f_mh) <= TOL)


# Test that we have same result as C++
from force_observer import ForceEstimator
force_estimator_cpp = ForceEstimator(robot.model, nc, nc, id_endeff, gains, pinRefFrame2)
# Assert default values
assert(force_estimator_cpp.frame_id == force_estimator.contact_frame_id)
assert(force_estimator_cpp.nc == force_estimator.nc)
assert(force_estimator_cpp.nc_delta_f == force_estimator.nc_delta_f)
assert(force_estimator_cpp.mask == force_estimator.mask)
assert(norm(force_estimator_cpp.H - force_estimator.H) <= TOL)
assert(norm(force_estimator_cpp.baumgarte_gains - force_estimator.baumgarte_gains) <= TOL)

# Assert estimation
force_estimator_data = force_estimator_cpp.createData()
force_estimator_cpp.estimate(force_estimator_data, q.copy(), v.copy(), a.copy(), tau.copy(), df.copy(), f.copy())
assert(norm(force_estimator_data.delta_f - delta_f) <= TOL)


print("OK !")



