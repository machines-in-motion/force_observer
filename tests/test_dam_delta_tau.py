'''
Test that the C++ class DAMContactDeltaTau matches Crocoddyl DAM
when delta_tau = 0
'''
import pinocchio as pin
import example_robot_data as robex
import numpy as np
from numpy.linalg import norm
np.set_printoptions(precision=4, linewidth=180)
import crocoddyl
from force_observer import DAMContactDeltaTau

nc = 3
pinRefFrame = pin.LOCAL_WORLD_ALIGNED

np.random.seed(1)

TOL = 1e-2


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

# Test cases
ROBOTS = ['talos_arm']
FRAMES = ['wrist_left_ft_tool_link']

FRAMES_REF_STR = ['LOCAL', 'LOCAL_WORLD_ALIGNED']
FRAMES_REF     = [pin.LOCAL, pin.LOCAL_WORLD_ALIGNED]

CONTACT_DIMS = [1, 3] # 6]

# Main testing loop

for _, nc in enumerate(CONTACT_DIMS):

    for idf,frame in enumerate(FRAMES_REF):
        pinRefFrameStr = FRAMES_REF_STR[idf]
        pinRefFrame    = frame

        for idr,robot_name in enumerate(ROBOTS):

            # Robot name and corresponding operational frame
            robot = robex.load(robot_name)
            contactFrameName = FRAMES[idr]
            # Initial conditions
            q0 = pin.randomConfiguration(robot.model) 
            v0 = np.random.rand(robot.model.nv) 
            a0 = np.random.rand(robot.model.nv) 
            tau0 = np.random.rand(robot.model.nv) 
            x0 = np.concatenate([q0, v0])
            f = np.random.rand(nc)
            df = np.random.rand(nc)
            nq = robot.model.nq
            nv = robot.model.nv
            id_endeff = robot.model.getFrameId(contactFrameName)

            print("\nTEST CASE "+robot_name+"_"+contactFrameName+"_"+pinRefFrameStr+"_NC="+str(nc))
            

            contact_frame_id = robot.model.getFrameId(contactFrameName)
            jMf = robot.model.frames[contact_frame_id].placement
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


            # Create DAM delta_tau and classical DAM
            DAM = DAMContactDeltaTau(state, actuation, contactModel, runningCostModel, 0., enable_force=True)
            DAD = DAM.createData()
            DAM_crocoddyl = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, runningCostModel, 0., enable_force=True)
            DAD_crocoddyl = DAM_crocoddyl.createData()

            # Check that DAM and DAM delta_tau are equivalent when delta_tau = 0
            DAM.delta_tau = np.zeros(nv) 
            DAM.calc(DAD, x0, tau0)
            DAM.calcDiff(DAD, x0, tau0)
            DAM_crocoddyl.calc(DAD_crocoddyl, x0, tau0)
            DAM_crocoddyl.calcDiff(DAD_crocoddyl, x0, tau0)
            assert(norm(DAD.xout - DAD_crocoddyl.xout) <= TOL)
            assert(norm(DAD.Fx - DAD_crocoddyl.Fx) <= TOL)
            assert(norm(DAD.Fu - DAD_crocoddyl.Fu) <= TOL)

            assert(norm(DAD.pinocchio.dtau_dq - DAD_crocoddyl.pinocchio.dtau_dq) <= TOL)
            assert(norm(DAD.df_dx - DAD_crocoddyl.df_dx) <= TOL)
            assert(norm(DAD.df_du - DAD_crocoddyl.df_du) <= TOL)

            # Check that they differ when delta_tau is not zero
            # DAM.delta_tau = np.random.rand(nv)
            DAM.calc(DAD, x0, tau0)
            # assert(norm(DAD.xout - DAD_crocoddyl.xout) > TOL)

            # Compute derivative of DAM delta_tau and check against NumDiff
            DAM.calcDiff(DAD, x0, tau0)


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


            # Joint acc derivatives
            ddq_dq = DAD.Fx[:,:nq]
            ddq_dv = DAD.Fx[:,nq:]
            ddq_dtau = DAD.Fu
            # Force derivatives
            if(nc > 1):
                df_dq = DAD.df_dx[:,:nq]
                df_dv = DAD.df_dx[:,nq:]
            else:
                df_dq = DAD.df_dx[:nq]
                df_dv = DAD.df_dx[nq:] 
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
            assert(norm(ddq_dq - ddq_dq_ND)<=TOL)
            assert(norm(ddq_dv - ddq_dv_ND)<=TOL)
            assert(norm(ddq_dtau - ddq_dtau_ND)<=TOL)
            if(nc > 1):
                assert(norm(df_dq[:nc] - df_dq_ND)<=TOL)      
                assert(norm(df_dv[:nc] - df_dv_ND)<=TOL)
                assert(norm(df_dtau[:nc] - df_dtau_ND)<=TOL)
            else:
                assert(norm(df_dq - df_dq_ND)<=TOL)      
                assert(norm(df_dv - df_dv_ND)<=TOL)
                assert(norm(df_dtau - df_dtau_ND)<=TOL) 
            if(pinRefFrame == pin.LOCAL):
                assert(norm(dtau_dq - dtau_dq_ND)<=TOL)   
                assert(norm(dtau_dv - dtau_dv_ND)<=TOL)   
