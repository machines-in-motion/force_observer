'''
Test python protoype of modified DAM 
 - numdiff derivatives 
 - equivalent with Crocoddyl DAM when delta_f = 0
'''
import pinocchio as pin
import example_robot_data as robex
import numpy as np
np.set_printoptions(precision=4, linewidth=180)
import crocoddyl

import pathlib
import os
python_path = pathlib.Path('.').absolute().parent.parent/'demos'
os.sys.path.insert(1, str(python_path))
print(python_path)

from ContactModel import DAMRigidContact
from ContactModel1d_3d import DAMRigidContact1D3D


nc = 1
nc_delta_f = 1
pinRefFrame = pin.LOCAL_WORLD_ALIGNED

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
nq = robot.model.nq
nv = robot.model.nv

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
contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)
# contact_position = robot.data.oMf[contact_frame_id].copy()
baumgarte_gains  = np.array([0., 0.])
if(nc == 3):
    contactItem = crocoddyl.ContactModel3D(state, contact_frame_id, robot.data.oMf[contact_frame_id].translation, pinRefFrame, actuation.nu, baumgarte_gains) 
elif(nc == 6):
    contactItem = crocoddyl.ContactModel6D(state, contact_frame_id, robot.data.oMf[contact_frame_id], pinRefFrame, actuation.nu, baumgarte_gains) 
elif(nc == 1 or nc == 0):
    M = pin.SE3.Identity()
    contactItem = crocoddyl.ContactModel1D(state, contact_frame_id, robot.data.oMf[contact_frame_id].translation[2], pinRefFrame, M.rotation, actuation.nu, baumgarte_gains) 
else: pass
# Populate contact model with contacts
if(nc == 0):
    contactModel.addContact("contact", contactItem, active=False)
else:
    contactModel.addContact("contact", contactItem, active=True)
# Create cost terms 
uResidual = crocoddyl.ResidualModelContactControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
xResidual = crocoddyl.ResidualModelState(state, x0 )
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
desired_wrench = np.array([0., 0., -100., 0., 0., 0.])
frameForceResidual = crocoddyl.ResidualModelContactForce(state, contact_frame_id, pin.Force(desired_wrench), nc, actuation.nu)
contactForceCost = crocoddyl.CostModelResidual(state, frameForceResidual)
runningCostModel.addCost("stateReg", xRegCost, 1e-2)
runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
runningCostModel.addCost("force", contactForceCost, 10.)
terminalCostModel.addCost("stateReg", xRegCost, 1e-2)
# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions

delta_f = 100*np.random.rand(nc_delta_f) 
if nc != nc_delta_f:
    DAM = DAMRigidContact1D3D(state, actuation, contactModel, runningCostModel, contact_frame_id, delta_f, pinRefFrame)
else:
    DAM = DAMRigidContact(state, actuation, contactModel, runningCostModel, contact_frame_id, delta_f, pinRefFrame)

DAD = DAM.createData()
jMf = robot.model.frames[contact_frame_id].placement

def ddq_dam(dam,dad, q, v, u):
    dam.calc(dad, np.concatenate([q,v]), u)
    return dad.xout

def df_dam(dam, dad, q, v, u):
    '''
    Careful: cd.fext contains the contact force (spatial) expressed at the parent joint level
    while DAD.df_dx is already expressed in the desired frame of reference
    So we need to be transform cd.fext through jMf^-1 (in LOCAL) or lwaMf*jMf^-1 (in LWA)
        Special case of contact 1D: there can be an additional rotation that
        rotates the z-axis of the reference frame in any arbitrary direction
    '''
    dam.calc(dad, np.concatenate([q,v]), u)
    cd = dad.multibody.contacts.contacts['contact']
    if(nc == 1):
        if(pinRefFrame == pin.LOCAL):
            return jMf.actInv(M.actInv(cd.fext)).vector[2]
        else:
            lwaMf = dad.pinocchio.oMf[contact_frame_id].copy() ; lwaMf.translation = np.zeros(3)
            return M.act(lwaMf.act(jMf.actInv(cd.fext))).vector[2]
    else:
        if(pinRefFrame == pin.LOCAL):
            return jMf.actInv(cd.fext).vector[:nc]
        else:
            lwaMf = dad.pinocchio.oMf[contact_frame_id].copy() ; lwaMf.translation = np.zeros(3)
            return lwaMf.act(jMf.actInv(cd.fext)).vector[:nc]

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
if(pinRefFrame == pin.LOCAL):
    dtau_dq_ND = numdiff(lambda q_:tau_dam(DAM, DAD, q_, v0, tau0), q0)
    dtau_dv_ND = numdiff(lambda v_:tau_dam(DAM, DAD, q0, v_, tau0), v0)

# Tests
assert(np.linalg.norm(ddq_dq - ddq_dq_ND)<=TOL)
assert(np.linalg.norm(ddq_dv - ddq_dv_ND)<=TOL)
assert(np.linalg.norm(ddq_dtau - ddq_dtau_ND)<=TOL)
assert(np.linalg.norm(df_dq[:nc] - df_dq_ND)<=TOL)      
assert(np.linalg.norm(df_dv[:nc] - df_dv_ND)<=TOL)
assert(np.linalg.norm(df_dtau[:nc] - df_dtau_ND)<=TOL)
if(pinRefFrame == pin.LOCAL):
    assert(np.linalg.norm(dtau_dq - dtau_dq_ND)<=TOL)   
    assert(np.linalg.norm(dtau_dv - dtau_dv_ND)<=TOL)   



# Check that our custom DAM is equivalent to croco DAM when delta_f = 0
DAM_croco = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, runningCostModel, inv_damping=0., enable_force=True)
DAD_croco = DAM_croco.createData()
DAM_croco.calc(DAD_croco, x0, tau0)
DAM_croco.calcDiff(DAD_croco, x0, tau0)


delta_f = np.zeros(nc_delta_f)
if nc != nc_delta_f:
    DAM2 = DAMRigidContact1D3D(state, actuation, contactModel, runningCostModel, contact_frame_id, delta_f, pinRefFrame)
else:
    DAM2 = DAMRigidContact(state, actuation, contactModel, runningCostModel, contact_frame_id, delta_f, pinRefFrame)
DAD2 = DAM2.createData()
DAM2.calc(DAD2, x0, tau0)
DAM2.calcDiff(DAD2, x0, tau0)

assert(np.linalg.norm(DAD_croco.xout - DAD2.xout)<=TOL)   
assert(np.linalg.norm(DAD_croco.cost - DAD2.cost)<=TOL)   
assert(np.linalg.norm(DAD_croco.pinocchio.lambda_c - DAD2.pinocchio.lambda_c)<=TOL)   
assert(np.linalg.norm(DAD_croco.Fx - DAD2.Fx)<=TOL)   
assert(np.linalg.norm(DAD_croco.Fu - DAD2.Fu)<=TOL)   
assert(np.linalg.norm(DAD_croco.Lx - DAD2.Lx)<=TOL)   
assert(np.linalg.norm(DAD_croco.Lu - DAD2.Lu)<=TOL)   
assert(np.linalg.norm(DAD_croco.Lxx - DAD2.Lxx)<=TOL)   
assert(np.linalg.norm(DAD_croco.Lxu - DAD2.Lxu)<=TOL)   
assert(np.linalg.norm(DAD_croco.Luu - DAD2.Luu)<=TOL)   
assert(np.linalg.norm(DAD_croco.df_dx - DAD2.df_dx[:nc])<=TOL)   
assert(np.linalg.norm(DAD_croco.df_du - DAD2.df_du[:nc])<=TOL)   