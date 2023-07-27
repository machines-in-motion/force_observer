'''
Force Estimator
'''
import numpy as np
import pinocchio as pin
np.set_printoptions(precision=4, linewidth=180)

import pinocchio as pin
import proxsuite
import time


class Estimator():
    '''
    Computes the forward dynamics under rigid contact model 6D + force estimate
    '''
    def __init__(self, pin_robot, nc, nc_delta_f, frameId, baumgarte_gains, pinRefRame='LOCAL'):


        self.pin_robot = pin_robot
        self.nc = nc
        self.nc_delta_f = nc_delta_f

        if pinRefRame == 'LOCAL':
            self.pinRefRame = pin.LOCAL
        elif pinRefRame == 'LOCAL_WORLD_ALIGNED':
            self.pinRefRame = pin.LOCAL_WORLD_ALIGNED
        else: 
            assert False 

        if(self.nc == 1):
            self.mask = 2
            assert nc_delta_f == 1 or nc_delta_f == 3
        else:
            self.nc_delta_f = self.nc
            assert nc == nc_delta_f

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
        pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q, v,np.zeros(self.nv)) # a ?
        pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)
        M = self.pin_robot.mass(q)
        h = self.pin_robot.nle(q, v)
        if(self.nc == 1):
            alpha0 = pin.getFrameAcceleration(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame).vector[self.mask:self.mask+1]
            nu = pin.getFrameVelocity(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame).vector[self.mask:self.mask+1]
            J1 = pin.getFrameJacobian(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame)[self.mask:self.mask+1]
        else:
            alpha0 = pin.getFrameAcceleration(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame).vector[:self.nc]
            nu = pin.getFrameVelocity(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame).vector[:self.nc]
            J1 = pin.getFrameJacobian(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame)[:self.nc]
        alpha0 -= self.baumgarte_gains[1] * nu

        if self.nc_delta_f == 3 and self.nc == 1:
            J2 = pin.getFrameJacobian(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame)[:self.nc_delta_f]
        else:
            J2 = J1
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



# Implementation with Df in constraint

# class MHEstimator():
#     '''
#     Computes the forward dynamics under rigid contact model 6D + force estimate
#     '''
#     def __init__(self, T, pin_robot, nc, frameId, baumgarte_gains, pinRefRame='LOCAL'):


#         self.pin_robot = pin_robot
#         self.nc = nc

#         if pinRefRame == 'LOCAL':
#             self.pinRefRame = pin.LOCAL
#         elif pinRefRame == 'LOCAL_WORLD_ALIGNED':
#             self.pinRefRame = pin.LOCAL_WORLD_ALIGNED
#         else: 
#             assert False 

#         if(self.nc == 1):
#             self.mask = 2


#         self.nv = self.pin_robot.model.nv

#         self.R = 1e-2 * np.eye(self.nc)
#         self.Q = 1e-2 * np.eye(self.nv)
#         self.P = 1e-0 * np.eye(self.nc)


#         self.n_tot = T * (self.nv + self.nc) + self.nc
#         n_eq = T * (self.nv + self.nc)
#         n_in = 0

#         self.qp = proxsuite.proxqp.sparse.QP(self.n_tot, n_eq, n_in)

#         self.contact_frame_id = frameId

#         self.A = np.zeros((n_eq, self.n_tot))
#         self.b = np.zeros(n_eq)

#         self.H = np.zeros((self.n_tot, self.n_tot))
#         for t in range(T):
#             ind =  t * (self.nv + self.nc)
#             self.H[ind:ind+self.nv, ind:ind+self.nv]  = self.Q
#             self.H[ind+self.nv:ind+self.nv+self.nc, ind+self.nv:ind+self.nv+self.nc]  = self.R 
#         self.H[-self.nc:, -self.nc:]  = self.P 

#         self.g = np.zeros(self.n_tot)


#         self.C = None
#         self.u = None
#         self.l = None

#         self.baumgarte_gains = baumgarte_gains

#         assert self.baumgarte_gains[0] == 0

#         self.T = T

#     def estimate(self, q_list, v_list, a_list, tau_list, df_prior, F_mes_list):
#         for t in range(self.T):
#             pin.computeAllTerms(self.pin_robot.model, self.pin_robot.data, q_list[t], v_list[t])
#             pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q_list[t], v_list[t], a_list[t]) #np.zeros(self.nv))
#             pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)
#             M = self.pin_robot.mass(q_list[t]).copy()
#             h = self.pin_robot.nle(q_list[t], v_list[t]).copy()
#             if(self.nc == 1):
#                 alpha0 = pin.getFrameAcceleration(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame).vector[self.mask:self.mask+1]
#                 nu = pin.getFrameVelocity(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame).vector[self.mask:self.mask+1]
#                 J = pin.getFrameJacobian(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame)[self.mask:self.mask+1]
#             else:
#                 alpha0 = pin.getFrameAcceleration(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame).vector[:self.nc]
#                 nu = pin.getFrameVelocity(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame).vector[:self.nc]
#                 J = pin.getFrameJacobian(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame)[:self.nc]
#             alpha0 -= self.baumgarte_gains[1] * nu

#             ind =  t * (self.nv + self.nc)

#             self.b[ind:ind+self.nv + self.nc] = np.concatenate([h - tau_list[t], -alpha0], axis=0).copy()


#             self.A[ind:ind+self.nv, ind:ind+self.nv] = - M.copy()
#             self.A[ind:ind+self.nv, ind+self.nv:ind+2*self.nv] = J.T.copy()
#             self.A[ind:ind+self.nv, -self.nc:] = J.T.copy()

#             self.A[ind+self.nv:ind+self.nv+self.nc, ind:ind+self.nv] = J.copy()

            
#             # import pdb; pdb.set_trace()
#             self.g[ind:ind+self.nv] = - self.Q @ a_list[t].copy()
#             self.g[ind+self.nv:ind+self.nv+self.nc] = - self.R @ F_mes_list[t].copy()
#         self.g[-self.nc:] = - self.P @ df_prior.copy()

    
#         # import pdb; pdb.set_trace()
        
#         self.qp.init(self.H, self.g, self.A, self.b, self.C, self.l, self.u)



#         t1 = time.time()

#         self.qp.solve()
#         # print("time ", time.time() - t1)
#         # print("iter ", self.qp.results.info.iter)
#         # print("primal ", self.qp.results.info.pri_res)
#         # print("dual ", self.qp.results.info.dua_res)
#         # print(1/0)


#         return None, self.qp.results.x[-self.nc:]



# Implementation with Df in cost


class MHEstimator():
    '''
    Computes the forward dynamics under rigid contact model 6D + force estimate
    '''
    def __init__(self, T, pin_robot, nc, frameId, baumgarte_gains, pinRefRame='LOCAL'):


        self.pin_robot = pin_robot
        self.nc = nc

        if pinRefRame == 'LOCAL':
            self.pinRefRame = pin.LOCAL
        elif pinRefRame == 'LOCAL_WORLD_ALIGNED':
            self.pinRefRame = pin.LOCAL_WORLD_ALIGNED
        else: 
            assert False 

        if(self.nc == 1):
            self.mask = 2


        self.nv = self.pin_robot.model.nv

        self.R = 1e-2 * np.eye(self.nc)
        self.Q = 1e-2 * np.eye(self.nv)
        self.P = 5e-1 * np.eye(self.nc)


        self.n_tot = T * (self.nv + self.nc) + self.nc
        n_eq = T * (self.nv + self.nc)
        n_in = 0

        self.qp = proxsuite.proxqp.sparse.QP(self.n_tot, n_eq, n_in)

        self.contact_frame_id = frameId

        self.A = np.zeros((n_eq, self.n_tot))
        self.b = np.zeros(n_eq)

        self.H = np.zeros((self.n_tot, self.n_tot))
        for t in range(T):
            ind =  t * (self.nv + self.nc)
            self.H[ind:ind+self.nv, ind:ind+self.nv]  = self.Q
            self.H[ind+self.nv:ind+self.nv+self.nc, ind+self.nv:ind+self.nv+self.nc]  = self.R 
            self.H[ind+self.nv+self.nc:ind+self.nv+2*self.nc, ind+self.nv+self.nc:ind+self.nv+2*self.nc]  = self.R 


            self.H[ind+self.nv+self.nc:ind+self.nv+2*self.nc, ind-self.nc:ind]  = self.R 
            self.H[ind+self.nv+self.nc:ind+self.nv+2*self.nc,  (t+2) * (self.nv + self.nc)-self.nc: (t+3) * (self.nv + self.nc)]  = self.R 



        self.H[-self.nc:, -self.nc:]  = self.P 

        self.g = np.zeros(self.n_tot)


        self.C = None
        self.u = None
        self.l = None

        self.baumgarte_gains = baumgarte_gains

        assert self.baumgarte_gains[0] == 0

        self.T = T

    def estimate(self, q_list, v_list, a_list, tau_list, df_prior, F_mes_list):
        for t in range(self.T):
            pin.computeAllTerms(self.pin_robot.model, self.pin_robot.data, q_list[t], v_list[t])
            pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q_list[t], v_list[t], np.zeros(self.nv))
            pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)
            M = self.pin_robot.mass(q_list[t]).copy()
            h = self.pin_robot.nle(q_list[t], v_list[t]).copy()
            if(self.nc == 1):
                alpha0 = pin.getFrameAcceleration(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame).vector[self.mask:self.mask+1]
                nu = pin.getFrameVelocity(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame).vector[self.mask:self.mask+1]
                J = pin.getFrameJacobian(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame)[self.mask:self.mask+1]
            else:
                alpha0 = pin.getFrameAcceleration(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame).vector[:self.nc]
                nu = pin.getFrameVelocity(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame).vector[:self.nc]
                J = pin.getFrameJacobian(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame)[:self.nc]
            alpha0 -= self.baumgarte_gains[1] * nu

            ind =  t * (self.nv + self.nc)

            self.b[ind:ind+self.nv + self.nc] = np.concatenate([h - tau_list[t], -alpha0], axis=0).copy()


            self.A[ind:ind+self.nv, ind:ind+self.nv] = - M.copy()
            self.A[ind:ind+self.nv, ind+self.nv:ind+2*self.nv] = J.T.copy()

            self.A[ind+self.nv:ind+self.nv+self.nc, ind:ind+self.nv] = J.copy()

            
            self.g[ind:ind+self.nv] = - self.Q @ a_list[t].copy()
            self.g[ind+self.nv:ind+self.nv+self.nc] = - self.R @ F_mes_list[t].copy()
            self.g[ind+self.nv+self.nc:ind+self.nv+2*self.nc] = - self.R @ F_mes_list[t].copy()
        self.g[-self.nc:] = - self.P @ df_prior.copy()

    
        self.qp.init(self.H, self.g, self.A, self.b, self.C, self.l, self.u)



        t1 = time.time()

        self.qp.solve()


        return None, self.qp.results.x[-self.nc:]



# Implementation with varying Delta F


class Varying_DF_MHEstimator():
    '''
    Computes the forward dynamics under rigid contact model 6D + force estimate
    '''
    def __init__(self, T, pin_robot, nc, frameId, baumgarte_gains, pinRefRame='LOCAL'):


        self.pin_robot = pin_robot
        self.nc = nc

        if pinRefRame == 'LOCAL':
            self.pinRefRame = pin.LOCAL
        elif pinRefRame == 'LOCAL_WORLD_ALIGNED':
            self.pinRefRame = pin.LOCAL_WORLD_ALIGNED
        else: 
            assert False 

        if(self.nc == 1):
            self.mask = 2


        self.nv = self.pin_robot.model.nv

        self.R = 1e-2 * np.eye(self.nc)
        self.Q = 1e-2 * np.eye(self.nv)
        self.P0 = 1e-8 * np.eye(self.nc)
        self.P = 1e1 * np.eye(self.nc)


        self.n_tot = T * (self.nv + self.nc + self.nc)
        n_eq = T * (self.nv + self.nc)
        n_in = 0

        self.qp = proxsuite.proxqp.sparse.QP(self.n_tot, n_eq, n_in)

        self.contact_frame_id = frameId

        self.A = np.zeros((n_eq, self.n_tot))
        self.b = np.zeros(n_eq)

        self.H = np.zeros((self.n_tot, self.n_tot))
        for t in range(T):
            ind =  t * (self.nv + 2*self.nc)
            self.H[ind:ind+self.nv, ind:ind+self.nv]  = self.Q
            self.H[ind+self.nv:ind+self.nv+self.nc, ind+self.nv:ind+self.nv+self.nc]  = self.R 
        
            self.H[(t+1) * (self.nv + 2*self.nc) - self.nc: (t + 1) * (self.nv + 2*self.nc), (t + 2) * (self.nv + 2*self.nc) - self.nc: (t + 2) * (self.nv + 2*self.nc)]  = - self.P 
            self.H[(t+1) * (self.nv + 2*self.nc) - self.nc: (t + 1) * (self.nv + 2*self.nc), t * (self.nv + 2*self.nc) - self.nc: t * (self.nv + 2*self.nc)]  = - self.P.T 
            # self.H[ind+self.nv+self.nc:ind+self.nv+2*self.nc, ind+self.nv+self.nc:ind+self.nv+2*self.nc]  = self.P 

        self.H[self.nv + self.nc: self.nv + 2*self.nc, self.nv + self.nc: self.nv + 2*self.nc]  = self.P0

        self.g = np.zeros(self.n_tot)


        self.C = None
        self.u = None
        self.l = None

        self.baumgarte_gains = baumgarte_gains

        assert self.baumgarte_gains[0] == 0

        self.T = T

    def estimate(self, q_list, v_list, a_list, tau_list, df_prior, F_mes_list):
        for t in range(self.T):
            pin.computeAllTerms(self.pin_robot.model, self.pin_robot.data, q_list[t], v_list[t])
            pin.forwardKinematics(self.pin_robot.model, self.pin_robot.data, q_list[t], v_list[t], np.zeros(self.nv)) # a_list[t]) #np.zeros(self.nv))
            pin.updateFramePlacements(self.pin_robot.model, self.pin_robot.data)
            M = self.pin_robot.mass(q_list[t])
            h = self.pin_robot.nle(q_list[t], v_list[t])
            if(self.nc == 1):
                alpha0 = pin.getFrameAcceleration(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame).vector[self.mask:self.mask+1]
                nu = pin.getFrameVelocity(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame).vector[self.mask:self.mask+1]
                J = pin.getFrameJacobian(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame)[self.mask:self.mask+1]
            else:
                alpha0 = pin.getFrameAcceleration(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame).vector[:self.nc]
                nu = pin.getFrameVelocity(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame).vector[:self.nc]
                J = pin.getFrameJacobian(self.pin_robot.model, self.pin_robot.data, self.contact_frame_id, self.pinRefRame)[:self.nc]
            alpha0 -= self.baumgarte_gains[1] * nu

            ind1 =  t * (self.nv + self.nc)
            ind2 =  t * (self.nv + 2*self.nc)

            self.b[ind1: ind1 + self.nv + self.nc] = np.concatenate([h - tau_list[t], -alpha0], axis=0)


            self.A[ind1:ind1+self.nv, ind2:ind2+self.nv] = - M 
            self.A[ind1:ind1+self.nv, ind2+self.nv:ind2+2*self.nv] = J.T 
            self.A[ind1:ind1+self.nv, ind2+2*self.nv:ind2+3*self.nv] = J.T 
            self.A[ind1+self.nv:ind1+self.nv+self.nc, ind2:ind2+self.nv] = J

            
            # import pdb; pdb.set_trace()
            self.g[ind2:ind2+self.nv] = - self.Q @ a_list[t]
            self.g[ind2+self.nv:ind2+self.nv+self.nc] = - self.R @ F_mes_list[t]

            # self.g[ind2+self.nv+self.nc:ind2+self.nv+2*self.nc] = - self.P @ df_prior   


        self.g[self.nv + self.nc:self.nv + 2*self.nc] = - self.P0 @ df_prior   

    
        
        self.qp.init(self.H, self.g, self.A, self.b, self.C, self.l, self.u)



        t1 = time.time()

        self.qp.solve()

        return None, self.qp.results.x[-self.nc:]
