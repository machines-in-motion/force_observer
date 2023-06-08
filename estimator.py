'''
Force Estimator
'''

import crocoddyl
import numpy as np
import pinocchio as pin
np.set_printoptions(precision=4, linewidth=180)
import pin_utils, mpc_utils

from bullet_utils.env import BulletEnvWithGround
from robot_properties_kuka.iiwaWrapper import IiwaRobot
import pybullet as p
import pinocchio as pin
import proxsuite
import time


class Estimator():
    '''
    Computes the forward dynamics under rigid contact model 6D + force estimate
    '''
    def __init__(self, pin_robot, nc, frameId, baumgarte_gains):

        self.pin_robot = pin_robot
        self.nc = nc
        if(self.nc == 1):
            self.nc_delta_f = 3
            self.mask = 2
        else:
            self.nc_delta_f = self.nc
        self.nv = self.pin_robot.model.nv

        self.R = 1e-2 * np.eye(self.nc)
        self.Q = 1e-2 * np.eye(self.nv)
        self.P = 1e0 * np.eye(self.nc_delta_f)

        self.n_tot = self.nv + self.nc + self.nc_delta_f 
        n_eq = self.nv + self.nc
        n_in = 0

        self.qp = proxsuite.proxqp.dense.QP(self.n_tot, n_eq, n_in)

        self.contact_frame_id = frameId

        self.H = np.zeros((self.n_tot, self.n_tot))
        self.H[:self.nv, :self.nv]  = self.Q
        self.H[self.nv:self.nv+self.nc, self.nv:self.nv+self.nc]  = self.R 
        self.H[-self.nc_delta_f:, -self.nc_delta_f:]  = self.P 

        self.C = None
        self.u = None
        self.l = None

        self.baumgarte_gains = baumgarte_gains

        assert self.baumgarte_gains[0] == 0

    def estimate(self, q, v, a, tau, df_prior, F_mes):
        pin.computeAllTerms(self.pin_robot.model, self.pin_robot.data, q, v)
        pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q, v, np.zeros(self.nv)) # a ?
        pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)
        M = self.pin_robot.mass(q)
        h = self.pin_robot.nle(q, v)
        if(self.nc == 1):
            alpha0 = pin.getFrameAcceleration(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, pin.LOCAL).vector[self.mask:self.mask+1]
            nu = pin.getFrameVelocity(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, pin.LOCAL).vector[self.mask:self.mask+1]
            J1 = pin.getFrameJacobian(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, pin.LOCAL)[self.mask:self.mask+1]
        else:
            alpha0 = pin.getFrameAcceleration(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, pin.LOCAL).vector[:self.nc]
            nu = pin.getFrameVelocity(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, pin.LOCAL).vector[:self.nc]
            J1 = pin.getFrameJacobian(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, pin.LOCAL)[:self.nc]
        alpha0 -= self.baumgarte_gains[1] * nu

        J2 = pin.getFrameJacobian(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, pin.LOCAL)[:self.nc_delta_f]

        b = np.concatenate([h - tau, -alpha0], axis=0)
        A1 = np.concatenate([-M, J1.T, J2.T], axis=1)
        A2 = np.concatenate([J1, np.zeros((self.nc, self.nc + self.nc_delta_f))], axis=1)
        # import pdb; pdb.set_trace()
        A = np.concatenate([A1, A2], axis=0)

        # n_eq = A.shape[0]

        g = np.zeros(self.n_tot)
        # import pdb; pdb.set_trace()
        g[:self.nv] = - self.Q @ a
        g[self.nv:self.nv+self.nc] = - self.R @ F_mes
        g[-self.nc_delta_f:] = - self.P @ df_prior

        # solve it
        
        self.qp.init(self.H, g, A, b, self.C, self.l, self.u)



        t1 = time.time()
        self.qp.solve()
        # print("time ", time.time() - t1)
        # print("iter ", self.qp.results.info.iter)
        # print("primal ", self.qp.results.info.pri_res)
        # print("dual ", self.qp.results.info.dua_res)



        # self.qp.settings.initial_guess = (
        #     proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        # )

        # # # print an optimal solution
        # # print("optimal x: {}".format(qp.results.x))
        # # print("optimal y: {}".format(qp.results.y))
        # # print("optimal z: {}".format(qp.results.z))


        # t1 = time.time()

        # self.qp.solve()
        # print("time ", time.time() - t1)
        # print("iter ", self.qp.results.info.iter)
        # print("primal ", self.qp.results.info.pri_res)
        # print("dual ", self.qp.results.info.dua_res)
        # print(1/0)

        return self.qp.results.x[self.nv:self.nv+self.nc], self.qp.results.x[-self.nc_delta_f:]




# class Estimator():
#     '''
#     Computes the forward dynamics under rigid contact model 6D + force estimate
#     '''
#     def __init__(self):
#         self.R = 1e-2 * np.eye(6)
#         self.P = 1e-0 * np.eye(6)

#         n = 12
#         n_eq = 7
#         n_in = 0

#         self.qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)


#         self.H = np.zeros((12, 12))
#         self.H[:6, :6]  = self.R 
#         self.H[6:, 6:]  = self.P 


#         self.n = 12

#         self.C = None
#         self.u = None
#         self.l = None

#     def estimate(self, pin_robot, q, v, a, tau, df_prior, F_mes):

#         M = pin_robot.mass(q)
#         contact_frame_id = pin_robot.model.getFrameId("contact")
#         J = pin_robot.computeFrameJacobian(q, contact_frame_id)
#         h = pin_robot.nle(q, v)


#         b = M @ a + h - tau

#         A = np.concatenate([J.T, J.T], axis=1)
        
#         print(np.linalg.matrix_rank(A))

#         # n_eq = A.shape[0]

#         g = np.zeros(12)
#         # import pdb; pdb.set_trace()
#         g[:6] = - self.R @ F_mes
#         g[6:] = - self.P @ df_prior

#         # solve it
        
#         self.qp.init(self.H, g, A, b, self.C, self.l, self.u)



#         t1 = time.time()
#         self.qp.solve()
#         print("time ", time.time() - t1)
#         print("iter ", self.qp.results.info.iter)
#         print("primal ", self.qp.results.info.pri_res)
#         print("dual ", self.qp.results.info.dua_res)



#         # self.qp.settings.initial_guess = (
#         #     proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
#         # )

#         # # # print an optimal solution
#         # # print("optimal x: {}".format(qp.results.x))
#         # # print("optimal y: {}".format(qp.results.y))
#         # # print("optimal z: {}".format(qp.results.z))


#         # t1 = time.time()

#         # self.qp.solve()
#         # print("time ", time.time() - t1)
#         # print("iter ", self.qp.results.info.iter)
#         # print("primal ", self.qp.results.info.pri_res)
#         # print("dual ", self.qp.results.info.dua_res)
#         # print(1/0)

#         return self.qp.results.x[:6], self.qp.results.x[6:]

