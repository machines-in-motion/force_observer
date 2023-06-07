import pinocchio as pin
import example_robot_data as robex
import numpy as np
np.set_printoptions(precision=4, linewidth=180)
import crocoddyl
from ContactModel import DAMRigidContact
import sobec



np.random.seed(10)

TOL = 1e-3


# Load robot : works with talos arm, not with kinova
robot_name = 'talos_arm'# 'kinova'  #'talos_arm'
contactFrameName = 'wrist_left_ft_tool_link'  #'j2s6s200_joint_finger_tip_1' # 'wrist_left_ft_tool_link'
robot = robex.load(robot_name)
robot.data = robot.model.createData()
# Initial conditions
q0 = pin.randomConfiguration(robot.model) 
v0 = np.random.rand(robot.model.nv) 
a0 = np.random.rand(robot.model.nv) 
tau0 = np.random.rand(robot.model.nv) 
x0 = np.concatenate([q0, v0])
# BG gains
gains = np.zeros(2)
nq = robot.model.nq
nv = robot.model.nv
nc = 1

# Numerical difference function
def numdiff(f,inX,h=1e-6):
    f0 = f(inX).copy()
    x = inX.copy()
    Fx = []
    for ix in range(len(x)):
        x[ix] += h
        Fx.append((f(x)-f0)/h)
        x[ix] = inX[ix]
    return np.array(Fx).T


contact_frame_id = robot.model.getFrameId(contactFrameName)
pin.forwardKinematics(robot.model, robot.data, q0, v0, a0)
pin.updateFramePlacements(robot.model, robot.data)


state = crocoddyl.StateMultibody(robot.model)
actuation = crocoddyl.ActuationModelFull(state)
# Running and terminal cost models
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)
# Contact model 
# Create 3D contact on the en-effector frame
contactModel = sobec.ContactModelMultiple(state, actuation.nu)
# contact_position = robot.data.oMf[contact_frame_id].copy()
baumgarte_gains  = np.array([0., 50.])
if(nc == 3):
    contactItem = sobec.ContactModel3D(state, contact_frame_id, robot.data.oMf[contact_frame_id].translation, baumgarte_gains, pin.LOCAL) 
elif(nc == 6):
    contactItem = sobec.ContactModel6D(state, contact_frame_id, robot.data.oMf[contact_frame_id], baumgarte_gains, pin.LOCAL) 
elif(nc == 1):
    contactItem = sobec.ContactModel1D(state, contact_frame_id, robot.data.oMf[contact_frame_id].translation, actuation.nu, baumgarte_gains, sobec.Vector3MaskType.z, pin.LOCAL) 
else: pass
# Populate contact model with contacts
contactModel.addContact("contact", contactItem, active=True)
# Create cost terms 
uResidual = crocoddyl.ResidualModelContactControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
xResidual = crocoddyl.ResidualModelState(state, x0 )
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
desired_wrench = np.array([0., 0., -100., 0., 0., 0.])
frameForceResidual = sobec.ResidualModelContactForce(state, contact_frame_id, pin.Force(desired_wrench), nc, actuation.nu)
contactForceCost = crocoddyl.CostModelResidual(state, frameForceResidual)
runningCostModel.addCost("stateReg", xRegCost, 1e-2)
runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
runningCostModel.addCost("force", contactForceCost, 10.)
terminalCostModel.addCost("stateReg", xRegCost, 1e-2)
# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
if(nc == 1):
    delta_f = 100*np.random.rand(3) # delta_f[2] = -0
else:
    delta_f = 100*np.random.rand(nc) # delta_f[2] = -0
DAM = DAMRigidContact(state, actuation, contactModel, runningCostModel, contact_frame_id, delta_f)
DAD = DAM.createData()
jMf = robot.model.frames[contact_frame_id].placement

def ddq_dam(dam,dad, q, v, u):
    dam.calc(dad, np.concatenate([q,v]), u)
    return dad.xout
def df_dam(dam, dad, q, v, u):
    dam.calc(dad, np.concatenate([q,v]), u)
    cd = dad.multibody.contacts.contacts['contact']
    if(nc == 1):
        return jMf.actInv(cd.f).vector[2]
    else:
        return jMf.actInv(cd.f).vector[:nc]
def tau_dam(dam, dad, q, v, u):
    pin.rnea(dam.pinocchio, dad.pinocchio, q, v, dad.xout, dad.multibody.contacts.fext)
    return dad.pinocchio.tau 

# relation between LOCAL and LWA derivatives
DAM.calc(DAD, x0, tau0)
DAM.calcDiff(DAD, x0, tau0)

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
ddq_dq_ND = numdiff(lambda q_:ddq_dam(DAM, DAD, q_, v0, tau0), q0)
ddq_dv_ND = numdiff(lambda v_:ddq_dam(DAM, DAD, q0, v_, tau0), v0)
ddq_dtau_ND = numdiff(lambda tau_:ddq_dam(DAM, DAD, q0, v0, tau_), tau0)
# Force derivative (ND)
df_dq_ND = numdiff(lambda q_:df_dam(DAM, DAD, q_, v0, tau0), q0)
df_dv_ND = numdiff(lambda v_:df_dam(DAM, DAD, q0, v_, tau0), v0)
df_dtau_ND = numdiff(lambda tau_:df_dam(DAM, DAD, q0, v0, tau_), tau0)
# RNEA derivatives (ND)
dtau_dq_ND = numdiff(lambda q_:tau_dam(DAM, DAD, q_, v0, tau0), q0)
dtau_dv_ND = numdiff(lambda v_:tau_dam(DAM, DAD, q0, v_, tau0), v0)

# Tests
assert(np.linalg.norm(ddq_dq - ddq_dq_ND)<=TOL)
assert(np.linalg.norm(ddq_dv - ddq_dv_ND)<=TOL)
assert(np.linalg.norm(ddq_dtau - ddq_dtau_ND)<=TOL)
assert(np.linalg.norm(df_dq[:nc] - df_dq_ND)<=TOL)      
assert(np.linalg.norm(df_dv[:nc] - df_dv_ND)<=TOL)
assert(np.linalg.norm(df_dtau[:nc] - df_dtau_ND)<=TOL)

assert(np.linalg.norm(dtau_dq - dtau_dq_ND)<=TOL)   
assert(np.linalg.norm(dtau_dv - dtau_dv_ND)<=TOL)   
