import pinocchio as pin
import example_robot_data as robex
import numpy as np
from numpy.linalg import norm
from numpy.random import rand 
np.set_printoptions(precision=4, linewidth=180)
import crocoddyl
from ContactModel import DAMRigidContact

np.random.seed(10)

TOL = 1e-3


# Load robot : works with talos arm, not with kinova
robot_name = 'talos_arm'# 'kinova'  #'talos_arm'
contactFrameName = 'wrist_left_ft_tool_link'  #'j2s6s200_joint_finger_tip_1' # 'wrist_left_ft_tool_link'
robot = robex.load(robot_name)
model = robot.model
# Initial conditions
q = pin.randomConfiguration(model) 
v = rand(model.nv) 
a = rand(model.nv) 
tau = rand(model.nv) 
x = np.concatenate([q, v])
# BG gains
gains = np.zeros(2)
nq = model.nq
nv = model.nv
nc = 6 #6


# Numerical difference function
def numdiff(f,x0,h=1e-6):
    f0 = f(x0).copy()
    x = x0.copy()
    Fx = []
    for ix in range(len(x)):
        x[ix] += h
        Fx.append((f(x)-f0)/h)
        x[ix] = x0[ix]
    return np.array(Fx).T


contact_frame_id = robot.model.getFrameId(contactFrameName)
# pin.forwardKinematics(robot.model, robot.data, q, v, a)
# pin.updateFramePlacements(robot.model, robot.data)


state = crocoddyl.StateMultibody(robot.model)
actuation = crocoddyl.ActuationModelFull(state)
# Running and terminal cost models
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)
# Contact model 
contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)
# Create 3D contact on the en-effector frame
# contact_position = robot.data.oMf[contact_frame_id].copy()
baumgarte_gains  = np.array([0., 50.])
if(nc == 3):
    contactItem = crocoddyl.ContactModel3D(state, contact_frame_id, robot.data.oMf[contact_frame_id].translation, baumgarte_gains) 
else:
    contactItem = crocoddyl.ContactModel6D(state, contact_frame_id, robot.data.oMf[contact_frame_id], baumgarte_gains) 
# Populate contact model with contacts
contactModel.addContact("contact", contactItem, active=True)
# Create cost terms 
uResidual = crocoddyl.ResidualModelContactControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
xResidual = crocoddyl.ResidualModelState(state, x)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
desired_wrench = np.array([0., 0., -100., 0., 0., 0.])
frameForceResidual = crocoddyl.ResidualModelContactForce(state, contact_frame_id, pin.Force(desired_wrench), nc, actuation.nu)
contactForceCost = crocoddyl.CostModelResidual(state, frameForceResidual)
runningCostModel.addCost("stateReg", xRegCost, 1e-2)
runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
runningCostModel.addCost("force", contactForceCost, 10.)
terminalCostModel.addCost("stateReg", xRegCost, 1e-2)
# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
delta_f = 100*np.random.rand(nc) # delta_f[2] = -0
DAM = DAMRigidContact(state, actuation, contactModel, runningCostModel, contact_frame_id, delta_f)
DAD = DAM.createData()
jMf = robot.model.frames[contact_frame_id].placement

def ddq_dam(dam,dad, q, v, u):
    x = np.concatenate([q,v])
    dam.calc(dad, x, u)
    return dad.xout
def df_dam(dam, dad, q, v, u):
    x = np.concatenate([q,v])
    dam.calc(dad, x, u)
    cd = dad.multibody.contacts.contacts['contact']
    return jMf.actInv(cd.f).vector[:nc]
def tau_dam(dam, dad, q, v, u):
    pin.rnea(dam.pinocchio, dad.pinocchio, q, v, dad.xout, dad.multibody.contacts.fext)
    return dad.pinocchio.tau 

# relation between LOCAL and LWA derivatives
DAM.calc(DAD, x, tau)
DAM.calcDiff(DAD, x, tau)

# Joint acc derivatives
ddq_dq = DAD.Fx[:,:nq]
ddq_dv = DAD.Fx[:,nq:]
ddq_dtau = DAD.Fu
# Force derivatives
df_dq = DAD.df_dx[:,:nq]
df_dv = DAD.df_dx[:,nq:]
df_dtau = DAD.df_du
# RNEA derivatives
dtau_dq = DAD.pinocchio.dtau_dq
dtau_dv = DAD.pinocchio.dtau_dv

# Joint acc derivatives (ND)
ddq_dq_ND = numdiff(lambda q_:ddq_dam(DAM, DAD, q_, v, tau), q)
ddq_dv_ND = numdiff(lambda v_:ddq_dam(DAM, DAD, q, v_, tau), v)
ddq_dtau_ND = numdiff(lambda tau_:ddq_dam(DAM, DAD, q, v, tau_), tau)
# Force derivative (ND)
df_dq_ND = numdiff(lambda q_:df_dam(DAM, DAD, q_, v, tau), q)
df_dv_ND = numdiff(lambda v_:df_dam(DAM, DAD, q, v_, tau), v)
df_dtau_ND = numdiff(lambda tau_:df_dam(DAM, DAD, q, v, tau_), tau)
# RNEA derivatives (ND)
dtau_dq_ND = numdiff(lambda q_:tau_dam(DAM, DAD, q_, v, tau), q)
dtau_dv_ND = numdiff(lambda v_:tau_dam(DAM, DAD, q, v_, tau), v)

# Tests
assert(norm(ddq_dq - ddq_dq_ND)<=TOL)
assert(norm(ddq_dv - ddq_dv_ND)<=TOL)
assert(norm(ddq_dtau - ddq_dtau_ND)<=TOL)
assert(norm(df_dq - df_dq_ND)<=TOL)      
assert(norm(df_dv - df_dv_ND)<=TOL)
assert(norm(df_dtau - df_dtau_ND)<=TOL)

assert(norm(dtau_dq - dtau_dq_ND)<=TOL)   
assert(norm(dtau_dv - dtau_dv_ND)<=TOL)   
