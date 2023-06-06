'''
DAM contact 6d with delta_f
'''
import numpy as np
np.set_printoptions(precision=6, linewidth=180, suppress=True)
np.random.seed(1)

import crocoddyl
import pinocchio as pin

class DAMRigidContact6D(crocoddyl.DifferentialActionModelAbstract):
    '''
    Computes the forward dynamics under rigid contact model 6D + force estimate
    '''
    def __init__(self, stateMultibody, actuationModel, contactModelSum, costModelSum, frameId, delta_f):
        # super(DAMSoftContactDynamics, self).__init__(stateMultibody, actuationModel.nu, costModelSum.nr)
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


        self.nc = 6
        self.nv = 7

    def createData(self):
        '''
            The data is created with a custom data class
        '''
        data = DADRigidContact6D(self)
        return data

    def calc(self, data, x, u=None):
        '''
        Compute joint acceleration and costs
        '''
        q = x[:self.state.nq]
        v = x[self.state.nq:]
        pin.computeAllTerms(self.pinocchio, data.pinocchio, q, v)
        pin.computeCentroidalMomentum(self.pinocchio, data.pinocchio)
        pin.forwardKinematics(self.pinocchio, data.pinocchio, q, v, np.zeros(self.state.nq))
        pin.updateFramePlacements(self.pinocchio, data.pinocchio)


        if u is None:
            print("hey")
            u = np.zeros(self.nu)

        # Actuation calc
        self.actuation.calc(data.multibody.actuation, x, u)
        self.contacts.calc(data.multibody.contacts, x)

        # Add delta_f
        new_tau = data.multibody.actuation.tau #+ data.multibody.contacts.Jc[:self.nc].T @ self.delta_f

        pin.forwardDynamics(self.pinocchio, data.pinocchio,
                                            u,
                                            data.multibody.contacts.Jc[:self.nc],
                                            data.multibody.contacts.a0[:self.nc],
                                            0.)
        data.xout = data.pinocchio.ddq
        self.contacts.updateAcceleration(data.multibody.contacts, data.pinocchio.ddq)
        self.contacts.updateForce(data.multibody.contacts, data.pinocchio.lambda_c - self.delta_f) # just for the cost computation !!!
        self.costs.calc(data.costs, x, u)
        self.contacts.updateForce(data.multibody.contacts, data.pinocchio.lambda_c) # for dynamics! 
        data.cost = data.costs.cost
        # pin.updateGlobalPlacements(self.pinocchio, data.pinocchio)
        return data.xout, data.cost

    def calcDiff(self, data, x, u=None):
        '''
        Compute derivatives
        '''
        # print(u)

        if u is None:
            print("hey")
            u = np.zeros(self.nu)

        # First call calc
        q = x[:self.state.nq]
        v = x[self.state.nq:]
        # Actuation calcDiff
        self.actuation.calcDiff(data.multibody.actuation, x, u)

        # This line makes unittest pass for somehow...
        # self.contacts.updateForce(data.multibody.contacts, data.pinocchio.lambda_c) 
        # data.pinocchio.tau -= data.multibody.contacts.Jc[:self.nc].T @ self.delta_f
        # dnew_tau_dx =   

        pin.computeRNEADerivatives(self.pinocchio, data.pinocchio, q, v, data.xout, data.multibody.contacts.fext)
        data.Kinv = pin.getKKTContactDynamicMatrixInverse(self.pinocchio, data.pinocchio, data.multibody.contacts.Jc[:self.nc])

        self.actuation.calcDiff(data.multibody.actuation, x, u)
        self.contacts.calcDiff(data.multibody.contacts, x)

        # a_partial_dtau = data.Kinv.topLeftCorner(self.nv, self.nv)
        # a_partial_da = data.Kinv.topRightCorner(self.nv, self.nc)
        # f_partial_dtau = data.Kinv.bottomLeftCorner(self.nc, self.nv)
        # f_partial_da = data.Kinv.bottomRightCorner(self.nc, self.nc)

        a_partial_dtau = data.Kinv[:self.nv, :self.nv]
        a_partial_da = data.Kinv[:self.nv, -self.nc:]
        f_partial_dtau = data.Kinv[-self.nc:, :self.nv]
        f_partial_da = data.Kinv[-self.nc:, -self.nc:]

        data.Fx[:,:self.nv]  = -a_partial_dtau @ data.pinocchio.dtau_dq ##
        data.Fx[:,-self.nv:] = -a_partial_dtau @ data.pinocchio.dtau_dv
        # import pdb; pdb.set_trace()
        data.Fx -= a_partial_da @ data.multibody.contacts.da0_dx[:self.nc]
        data.Fx += a_partial_dtau @ data.multibody.actuation.dtau_dx
        # import pdb; pdb.set_trace()
        data.Fu = a_partial_dtau @ data.multibody.actuation.dtau_du

        data.df_dx[:self.nc, :self.nv] = f_partial_dtau @ data.pinocchio.dtau_dq ##
        data.df_dx[:self.nc, -self.nv:] = f_partial_dtau @ data.pinocchio.dtau_dv
        data.df_dx[:self.nc,:] += f_partial_da @ data.multibody.contacts.da0_dx[:self.nc]
        data.df_dx[:self.nc,:] -= f_partial_dtau @ data.multibody.actuation.dtau_dx
        data.df_du[:self.nc,:] = -f_partial_dtau @ data.multibody.actuation.dtau_du
        # self.contacts.updateAccelerationDiff(data.multibody.contacts, data.Fx[-self.nv:])
        self.contacts.updateAccelerationDiff(data.multibody.contacts, data.Fx)
        self.contacts.updateForceDiff(data.multibody.contacts, data.df_dx[:self.nc], data.df_du[:self.nc])
        self.costs.calcDiff(data.costs, x, u)
        data.Lx = data.costs.Lx
        data.Lu = data.costs.Lu
        data.Lxx = data.costs.Lxx
        data.Lxu = data.costs.Lxu
        data.Luu = data.costs.Luu


class DADRigidContact6D(crocoddyl.DifferentialActionDataAbstract):
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

        self.df_dx = np.zeros((am.nc, am.state.nx))  
        self.df_du = np.zeros((am.nc, am.actuation.nu))  
        self.Kinv  = np.zeros((am.state.nv + am.nc, am.state.nv + am.nc))
        self.pinocchio  = am.pinocchio.createData()
        self.actuation_data = am.actuation.createData()
        self.contact_data = am.contacts.createData(self.pinocchio)
        # self.multibody = crocoddyl.DataCollectorActMultibody(self.pinocchio, self.actuation_data)
        self.multibody = crocoddyl.DataCollectorActMultibodyInContact(self.pinocchio, self.actuation_data, self.contact_data)
        # self.costs = am.costs.createData(crocoddyl.DataCollectorMultibody(self.pinocchio))
        self.costs = am.costs.createData(crocoddyl.DataCollectorActMultibodyInContact(self.pinocchio, self.actuation_data, self.contact_data))