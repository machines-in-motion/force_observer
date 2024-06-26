'''
DAM contact 6d with delta_f
'''
import numpy as np
np.set_printoptions(precision=6, linewidth=180, suppress=True)
# np.random.seed(1)

from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

import crocoddyl
import pinocchio as pin

class DAMRigidContact1D3D(crocoddyl.DifferentialActionModelAbstract):
    '''
    Computes the forward dynamics under rigid contact model + force estimate
    '''
    def __init__(self, stateMultibody, actuationModel, contactModelSum, costModelSum, frameId, delta_f, pinRefFrame=pin.LOCAL):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, stateMultibody, actuationModel.nu, costModelSum.nr)
        self.frameId = frameId
        self.with_armature = False
        # To complete DAMAbstract into sth like DAMFwdDyn
        self.actuation = actuationModel
        self.costs = costModelSum
        self.contacts = contactModelSum
        self.pinocchio = stateMultibody.pinocchio
        self.delta_f = delta_f

        self.parentId = self.pinocchio.frames[self.frameId].parent
        self.jMf = self.pinocchio.frames[self.frameId].placement

        self.pinRefFrame = pinRefFrame

        self.nc = contactModelSum.nc
        self.nv = self.pinocchio.nv
        self.nq = self.pinocchio.nq
        self.nx = self.nv + self.nq
        self.ndx = stateMultibody.ndx
        if(self.nc == 1):
            self.J3d = np.zeros((3, self.pinocchio.nv))

    def createData(self):
        '''
            The data is created with a custom data class
        '''
        data = DADRigidContact(self)
        return data

    def calc(self, data, x, u=None):
        '''
        Compute joint acceleration and costs
        '''
        self.nc = self.contacts.nc 

        q = x[:self.state.nq]
        v = x[self.state.nq:]
        pin.computeAllTerms(self.pinocchio, data.pinocchio, q, v)
        pin.computeCentroidalMomentum(self.pinocchio, data.pinocchio)
        pin.forwardKinematics(self.pinocchio, data.pinocchio, q, v, np.zeros(self.state.nq))
        pin.updateFramePlacements(self.pinocchio, data.pinocchio)


        if u is None:
            u = np.zeros(self.nu)

        # Actuation calc
        self.actuation.calc(data.multibody.actuation, x, u)
        self.contacts.calc(data.multibody.contacts, x)

        # Add delta_f in torque space
        if(self.nc == 0):
            pin.aba(self.pinocchio, data.pinocchio, q, v, data.multibody.actuation.tau)
            pin.updateGlobalPlacements(self.pinocchio, data.pinocchio)
        if(self.nc == 1):
            self.J3d = pin.getFrameJacobian(self.pinocchio, data.pinocchio, self.frameId, self.pinRefFrame)[:3]
            new_tau = data.multibody.actuation.tau + self.J3d.T @ self.delta_f # Need 3D jac 
            pin.forwardDynamics(self.pinocchio, data.pinocchio,
                                                new_tau, #u,
                                                data.multibody.contacts.Jc.reshape((1, self.nv)), # 1D
                                                data.multibody.contacts.a0, # 1D
                                                0.)
        if(self.nc == 3 or self.nc == 6):
            new_tau = data.multibody.actuation.tau + data.multibody.contacts.Jc[:self.nc].T @ self.delta_f[:self.nc] 
            pin.forwardDynamics(self.pinocchio, data.pinocchio,
                                                new_tau, #u,
                                                data.multibody.contacts.Jc[:self.nc], 
                                                data.multibody.contacts.a0[:self.nc], 
                                                0.)
        data.xout = data.pinocchio.ddq
        self.contacts.updateAcceleration(data.multibody.contacts, data.pinocchio.ddq) 
        # Here we compute the cost without delta f
        if(self.nc != 0):
            self.contacts.updateForce(data.multibody.contacts, data.pinocchio.lambda_c)
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost
        # Here we add again delta_f to the computed force for dynamics
        if(self.nc == 1):
            if(self.pinRefFrame != pin.LOCAL):
                lwaMf = data.pinocchio.oMf[self.frameId] ; lwaMf.translation = np.zeros(3)
                self.contacts.updateForce(data.multibody.contacts, data.pinocchio.lambda_c) 
                data.multibody.contacts.fext[self.parentId] += self.jMf.act(pin.Force(np.concatenate([ lwaMf.rotation.T @ self.delta_f, np.zeros(3)])))
                data.multibody.contacts.contacts['contact'].f += self.jMf.act(pin.Force(np.concatenate([ lwaMf.rotation.T @ self.delta_f, np.zeros(3)])))

            else:
                self.contacts.updateForce(data.multibody.contacts, data.pinocchio.lambda_c) 
                data.multibody.contacts.fext[self.parentId] += self.jMf.act(pin.Force(np.concatenate([ self.delta_f, np.zeros(3)])))
                data.multibody.contacts.contacts['contact'].f += self.jMf.act(pin.Force(np.concatenate([ self.delta_f, np.zeros(3)])))
        else:
            if(self.nc != 0):
                self.contacts.updateForce(data.multibody.contacts, data.pinocchio.lambda_c + self.delta_f)    # 3D with (0,0,lambda_c) + delta_f
        # pin.updateGlobalPlacements(self.pinocchio, data.pinocchio)
        return data.xout, data.cost

    def calcDiff(self, data, x, u=None):
        '''
        Compute derivatives
        '''

        if u is None:
            u = np.zeros(self.nu)

        q = x[:self.state.nq]
        v = x[self.state.nq:]

        # Actuation calcDiff
        self.actuation.calcDiff(data.multibody.actuation, x, u)
        
        # Free fwd dynamics
        if(self.nc == 0):
            aba_dq, aba_dv, aba_dtau = pin.computeABADerivatives(self.pinocchio, data.pinocchio, q, v, data.multibody.actuation.tau)
            assert(np.linalg.norm(aba_dtau - data.pinocchio.Minv) <= 1e-8)
            data.Fx[:,:self.nv]  = aba_dq
            data.Fx[:,-self.nv:] = aba_dv
            data.Fx += data.pinocchio.Minv @ data.multibody.actuation.dtau_dx
            data.Fu = data.pinocchio.Minv @ data.multibody.actuation.dtau_du
        
        # Constrained dynamics
        else: 
            pin.computeRNEADerivatives(self.pinocchio, data.pinocchio, q, v, data.xout, data.multibody.contacts.fext) # 3D derivative with fext including delta_f
            if(self.nc == 1):
                data.Kinv = pin.getKKTContactDynamicMatrixInverse(self.pinocchio, data.pinocchio, data.multibody.contacts.Jc.reshape((1, self.nv))) # Kinv has nc=1 
            else:
                data.Kinv = pin.getKKTContactDynamicMatrixInverse(self.pinocchio, data.pinocchio, data.multibody.contacts.Jc[:self.nc])  

            # Contact derivatives
            self.contacts.calcDiff(data.multibody.contacts, x) 

            # Compute skew term to be added to rnea derivatives if we are in LWA 
            if(self.pinRefFrame == pin.LOCAL_WORLD_ALIGNED):
                data.pinocchio.dtau_dq[:self.nv, :self.nv] += data.multibody.contacts.contacts['contact'].dtau_dq 
                # missing skew term for delta_f 
                # pin.skew(data.pinocchio.oMf[self.frameId].rotation.T[:,2] * data.pinocchio.lambda_c)
                if(self.nc == 1):
                    tmp_skew = pin.skew(data.pinocchio.oMf[self.frameId].rotation.T @ self.delta_f) 
                    lJ3d = pin.getFrameJacobian(self.pinocchio, data.pinocchio, self.frameId, pin.LOCAL)
                    drnea_skew_term = -lJ3d[:3].T @ tmp_skew @ lJ3d[3:]
                    data.pinocchio.dtau_dq[:self.nv, :self.nv] += drnea_skew_term

            # Extract Kinv blocks
            a_partial_dtau = data.Kinv[:self.nv, :self.nv]
            a_partial_da = data.Kinv[:self.nv, -self.nc:]
            f_partial_dtau = data.Kinv[-self.nc:, :self.nv]
            f_partial_da = data.Kinv[-self.nc:, -self.nc:]

            # DAM partials
            data.Fx[:,:self.nv]  = -a_partial_dtau @ data.pinocchio.dtau_dq 
            data.Fx[:,-self.nv:] = -a_partial_dtau @ data.pinocchio.dtau_dv
            if(self.nc == 1):
                data.Fx -= a_partial_da @ data.multibody.contacts.da0_dx.reshape((1,self.ndx)) # 1D
            if(self.nc == 3 or self.nc == 6):
                data.Fx -= a_partial_da @ data.multibody.contacts.da0_dx[:self.nc]
            data.Fx += a_partial_dtau @ data.multibody.actuation.dtau_dx
            data.Fu = a_partial_dtau @ data.multibody.actuation.dtau_du

            # Update contact & force partials
            self.contacts.updateAccelerationDiff(data.multibody.contacts, data.Fx)
            data.df_dx[:self.nc, :self.nv] = f_partial_dtau @ data.pinocchio.dtau_dq 
            data.df_dx[:self.nc, -self.nv:] = f_partial_dtau @ data.pinocchio.dtau_dv
            if(self.nc == 1):
                data.df_dx[:self.nc,:] += f_partial_da @ data.multibody.contacts.da0_dx.reshape((1,self.ndx)) 
            if(self.nc == 3 or self.nc == 6):
                data.df_dx[:self.nc,:] += f_partial_da @ data.multibody.contacts.da0_dx[:self.nc]
            data.df_dx[:self.nc,:] -= f_partial_dtau @ data.multibody.actuation.dtau_dx
            data.df_du[:self.nc,:] = -f_partial_dtau @ data.multibody.actuation.dtau_du
            self.contacts.updateForceDiff(data.multibody.contacts, data.df_dx[:self.nc], data.df_du[:self.nc])
        
        # Cost partials        
        self.costs.calcDiff(data.costs, x, u)
        data.Lx = data.costs.Lx
        data.Lu = data.costs.Lu
        data.Lxx = data.costs.Lxx
        data.Lxu = data.costs.Lxu
        data.Luu = data.costs.Luu


class DADRigidContact(crocoddyl.DifferentialActionDataAbstract):
    '''
    Creates a data class with differential and augmented matrices from IAM (initialized with stateVector)
    '''
    def __init__(self, am):
        # super().__init__(am)
        crocoddyl.DifferentialActionDataAbstract.__init__(self, am)
        self.Fx = np.zeros((am.state.nq, am.state.nx))
        self.Fu = np.zeros((am.state.nq, am.nu))
        self.Lx = np.zeros(am.state.nx)
        self.Lu = np.zeros(am.actuation.nu)
        self.Lxx = np.zeros((am.state.nx, am.state.nx))
        self.Lxu = np.zeros((am.state.nx, am.actuation.nu))
        self.Luu = np.zeros((am.actuation.nu, am.actuation.nu))

        self.df_dx = np.zeros((6, am.state.nx))  
        self.df_du = np.zeros((6, am.actuation.nu))  
        self.Kinv  = np.zeros((am.state.nv + am.nc, am.state.nv + am.nc))
        self.pinocchio  = am.pinocchio.createData()
        self.actuation_data = am.actuation.createData()
        self.contact_data = am.contacts.createData(self.pinocchio)
        self.multibody = crocoddyl.DataCollectorActMultibodyInContact(self.pinocchio, self.actuation_data, self.contact_data)
        self.costs = am.costs.createData(crocoddyl.DataCollectorActMultibodyInContact(self.pinocchio, self.actuation_data, self.contact_data))