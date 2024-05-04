import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal as mvn
from tqdm import tqdm 
import time


#####################################################
# Agent
#####################################################
class Agent:
    def __init__(self, dt, dim):
        self.dt = dt
        self.dim = dim

    def dyn(self, st, ut):
        return ut
    
    def step(self, st, ut):
        st_new = st + self.dt * self.dyn(st, ut)
        return st_new

    def get_A(self, st, ut):
        return np.zeros((self.dim, self.dim))
    
    def get_B(self, st, ut):
        return np.eye(self.dim)
    
    def traj_sim(self, s0, u_traj):
        assert len(s0) == self.dim
        
        tsteps = len(u_traj)
        traj = np.zeros((tsteps, self.dim)) 
        st = s0.copy()
        for t in range(0, tsteps):
            ut = u_traj[t]
            st = self.step(st, ut)
            traj[t] = st.copy()

        return traj


#####################################################
# GMM
#####################################################
class GMM:
    def __init__(self, means, covs, ws):
        assert len(means) == len(covs)
        assert len(means) == len(ws)
        
        self.nmix = len(means)
        self.dim = len(means[0])

        self.means = np.array(means)
        self.covs = np.array(covs)
        self.ws = np.array(ws)

        self.covs_inv = []
        for _cov in self.covs:
            self.covs_inv.append(np.linalg.inv(_cov))
        self.covs_inv = np.array(self.covs_inv)

        self.norm = self.get_norm()
    
    def get_norm(self):
        norm_val = 0.0

        for i in range(self.nmix):
            for j in range(self.nmix):
                mean_i = self.means[i]
                mean_j = self.means[j]
                cov_i = self.covs[i]
                cov_j = self.covs[j]
                w_i = self.ws[i]
                w_j = self.ws[j]
                
                prefix = 1.0 / np.sqrt(np.linalg.det(2 * np.pi * (cov_i + cov_j))) * w_i * w_j
                norm_val_ij = prefix * np.exp(-0.5 * np.linalg.inv(cov_i + cov_j) @ (mean_i - mean_j) @ (mean_i - mean_j)) 
                norm_val += norm_val_ij
        
        return norm_val
    
    def pdf(self, x):
        val = 0.0
        for i in range(self.nmix):
            val += mvn.pdf(x, self.means[i], self.covs[i]) * self.ws[i]
        return val 
    
    def dpdf(self, x):
        dvec = 0.0
        for i in range(self.nmix):
            dvec += -1.0 * (self.covs_inv[i] @ (x - self.means[i])) * mvn.pdf(x, self.means[i], self.covs[i]) * self.ws[i]
        return dvec
    

#####################################################
# Kernel
#####################################################
class Kernel:
    def __init__(self, bw, dim):
        self.bw = bw 
        self.dim = dim
        self.cov = np.eye(self.dim) / self.bw
        self.cov_inv = np.eye(self.dim) * self.bw

    def eval(self, x1, x2):
        return mvn.pdf(x1, x2, self.cov)
    
    def grad_x1(self, x1, x2):
        return -1.0 * self.eval(x1, x2) * (self.cov_inv @ (x1-x2).T)
    
    def grad_x2(self, x1, x2):
        return +1.0 * self.eval(x1, x2) * (self.cov_inv @ (x1-x2).T)


#####################################################
# Barrier function
#####################################################
class Barr:
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max 

    def eval(self, x):
        val = 0.0
        val += np.sum((x > self.x_max) * np.square(x - self.x_max))
        val += np.sum((x < self.x_min) * np.square(self.x_min - x))
        return val 
    
    def grad(self, x):
        dvec = 0.0
        dvec += 2.0 * (x > self.x_max) * (x - self.x_max)
        dvec += 2.0 * (x < self.x_min) * (x - self.x_min)
        return dvec 
    

#####################################################
# Controller
#####################################################
class KES_iLQR:
    def __init__(self, 
                 tsteps: int, 
                 dt: float, 
                 dim: int, 
                 agent: Agent,
                 tgt: GMM, 
                 kernel: Kernel, 
                 barr: Barr, 
                 Q: np.ndarray, 
                 R: np.ndarray,
                 ucost_w: float,
                 barr_w: float) -> None:
        self.tsteps = tsteps
        self.dt = dt
        self.dim = dim 
        self.ucost_w = ucost_w
        self.barr_w = barr_w

        self.agent = agent
        self.tgt = tgt 
        self.kernel = kernel 
        self.barr = barr 

        self.info_w = 1.0
        self.explr_w = 1.0
        
        self.Q = Q
        self.Q_inv = np.linalg.inv(Q)
        self.R = R 
        self.R_inv = np.linalg.inv(R)

        self.s_traj = np.zeros((tsteps, dim))
        self.T = self.tsteps * self.dt 

    def loss(self, s_traj, u_traj):
        assert len(s_traj) == self.tsteps

        info_gain = self.dt * np.sum(
            self.tgt.pdf(s_traj)
        )
        info_gain *= 2.0 / self.T
        info_gain *= self.info_w

        explr_gain = self.dt * self.dt * np.sum(
            self.kernel.eval(s_traj - s_traj[:,np.newaxis], np.zeros(self.dim))
        )
        explr_gain *= 1.0 / np.power(self.T, 2)
        explr_gain *= self.explr_w

        barr_gain = self.dt * np.sum(self.barr.eval(s_traj))
        barr_gain *= self.barr_w

        ctrl_gain = self.dt * np.sum(np.square(u_traj)) * self.ucost_w

        return -1.0 * info_gain + explr_gain + barr_gain + ctrl_gain + self.tgt.norm
    
    def dldx(self, st, ut, s_traj):
        assert len(st) == self.dim 

        info_dvec = self.tgt.dpdf(st)
        info_dvec *= 2.0 / self.T 
        info_dvec *= self.info_w

        explr_dvec = self.dt * np.sum(self.kernel.grad_x2(s_traj, st), axis=1)
        explr_dvec *= 2.0 / np.power(self.T, 2)
        explr_dvec *= self.explr_w

        barr_dvec = self.barr.grad(st)
        barr_dvec *= self.barr_w

        # print(-1.0 * info_dvec, explr_dvec)

        return -1.0 * info_dvec + explr_dvec + barr_dvec
    
    def dldu(self, st, ut, s_traj):
        return 2.0 * ut * self.ucost_w
    
    def set_explr_w(self, w):
        self.explr_w = w

    def set_info_w(self, w):
        self.info_w = w
    
    def P_dyn_rev(self, Pt, At, Bt, at, bt):
        return Pt @ At + At.T @ Pt - Pt @ Bt @ self.R_inv @ Bt.T @ Pt + self.Q 
    
    def P_dyn_step(self, Pt, At, Bt, at, bt):
        k1 = self.dt * self.P_dyn_rev(Pt, At, Bt, at, bt)
        k2 = self.dt * self.P_dyn_rev(Pt+k1/2, At, Bt, at, bt)
        k3 = self.dt * self.P_dyn_rev(Pt+k2/2, At, Bt, at, bt)
        k4 = self.dt * self.P_dyn_rev(Pt+k3, At, Bt, at, bt)

        Pt_new = Pt + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0 
        return Pt_new 
    
    def P_traj_revsim(self, PT, A_list, B_list, a_list, b_list):
        P_traj_rev = np.zeros((self.tsteps, self.dim, self.dim))
        P_curr = PT.copy()
        for t in range(self.tsteps):
            At = A_list[-1-t]
            Bt = B_list[-1-t]
            at = a_list[-1-t]
            bt = b_list[-1-t]

            P_new = self.P_dyn_step(P_curr, At, Bt, at, bt)
            P_traj_rev[t] = P_new.copy()
            P_curr = P_new 
        
        return P_traj_rev

    def r_dyn_rev(self, rt, Pt, At, Bt, at, bt):
        return (At - Bt @ self.R_inv @ Bt.T @ Pt).T @ rt + at - Pt @ Bt @ self.R_inv @ bt

    def r_dyn_step(self, rt, Pt, At, Bt, at, bt):
        k1 = self.dt * self.r_dyn_rev(rt, Pt, At, Bt, at, bt)
        k2 = self.dt * self.r_dyn_rev(rt+k1/2, Pt, At, Bt, at, bt)
        k3 = self.dt * self.r_dyn_rev(rt+k2/2, Pt, At, Bt, at, bt)
        k4 = self.dt * self.r_dyn_rev(rt+k3, Pt, At, Bt, at, bt)

        rt_new = rt + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0 
        return rt_new
    
    def r_traj_revsim(self, rT, P_list, A_list, B_list, a_list, b_list):
        r_traj_rev = np.zeros((self.tsteps, self.dim))
        # r_curr = np.zeros(self.dim)
        r_curr = rT
        for t in range(self.tsteps):
            Pt = P_list[-1-t]
            At = A_list[-1-t]
            Bt = B_list[-1-t]
            at = a_list[-1-t]
            bt = b_list[-1-t]

            r_new = self.r_dyn_step(r_curr, Pt, At, Bt, at, bt)
            r_traj_rev[t] = r_new.copy()
            r_curr = r_new 

        return r_traj_rev

    def z_dyn(self, zt, Pt, rt, At, Bt, bt):
        return At @ zt + Bt @ self.z2v(zt, Pt, rt, Bt, bt)
    
    def z_dyn_step(self, zt, Pt, rt, At, Bt, bt):
        k1 = self.dt * self.z_dyn(zt, Pt, rt, At, Bt, bt)
        k2 = self.dt * self.z_dyn(zt+k1/2, Pt, rt, At, Bt, bt)
        k3 = self.dt * self.z_dyn(zt+k2/2, Pt, rt, At, Bt, bt)
        k4 = self.dt * self.z_dyn(zt+k3, Pt, rt, At, Bt, bt)

        zt_new = zt + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0 
        return zt_new

    def z_traj_sim(self, z0, P_list, r_list, A_list, B_list, b_list):
        z_traj = np.zeros((self.tsteps, self.dim))
        z_curr = z0.copy()

        for t in range(self.tsteps):
            Pt = P_list[t]
            rt = r_list[t]
            At = A_list[t]
            Bt = B_list[t]
            bt = b_list[t]

            z_new = self.z_dyn_step(z_curr, Pt, rt, At, Bt, bt)
            z_traj[t] = z_new.copy()
            z_curr = z_new
        
        return z_traj
    
    def z2v(self, zt, Pt, rt, Bt, bt):
        return -self.R_inv @ Bt.T @ Pt @ zt - self.R_inv @ Bt.T @ rt - self.R_inv @ bt
    
    def get_v_traj(self, s0, u_traj):
        s_traj = self.agent.traj_sim(s0, u_traj)

        A_list = np.zeros((self.tsteps, self.dim, self.dim))
        B_list = np.zeros((self.tsteps, self.dim, 2))
        a_list = np.zeros((self.tsteps, self.dim))
        b_list = np.zeros((self.tsteps, 2))
        for _i, (_st, _ut) in enumerate(zip(s_traj, u_traj)):
            A_list[_i] = self.agent.get_A(_st, _ut)
            B_list[_i] = self.agent.get_B(_st, _ut)
            a_list[_i] = self.dldx(_st, _ut, s_traj)
            b_list[_i] = self.dldu(_st, _ut, s_traj)

            # print(_i, a_list[_i])

        PT = np.zeros((self.dim, self.dim))
        P_traj_rev = self.P_traj_revsim(PT, A_list, B_list, a_list, b_list)
        P_list = np.flip(P_traj_rev, axis=0)

        rT = np.zeros(self.dim)
        r_traj_rev = self.r_traj_revsim(rT, P_list, A_list, B_list, a_list, b_list)
        r_list = np.flip(r_traj_rev, axis=0)

        # z0 = -1.0 * np.linalg.inv(P_list[0]) @ r_list[0]
        z0 = np.zeros(self.dim)
        z_list = self.z_traj_sim(z0, P_list, r_list, A_list, B_list, b_list)

        v_list = np.zeros((self.tsteps, 2))
        for t in range(self.tsteps):
            zt = z_list[t]
            Pt = P_list[t]
            rt = r_list[t]
            Bt = B_list[t]
            bt = b_list[t]
            v_list[t] = self.z2v(zt, Pt, rt, Bt, bt)

        return v_list
    
    def line_search(self, s0, u_traj, v_traj):
        s_traj = self.agent.traj_sim(s0, u_traj)
        opt_loss = self.loss(s_traj, u_traj)

        step_list = 1.0 * np.power(0.5, np.arange(10))
        opt_step = 0.0
        opt_s_traj = s_traj
        for step in step_list:
            temp_u_traj = u_traj + step * v_traj
            temp_s_traj = self.agent.traj_sim(s0, temp_u_traj)
            temp_loss = self.loss(temp_s_traj, temp_u_traj)
            if temp_loss < opt_loss:
                opt_loss = temp_loss
                opt_step = step
                opt_s_traj = temp_s_traj

        return opt_step, opt_s_traj, opt_loss
        

def main():
    #################################################
    # Global parameters
    #################################################
    dt = 0.1
    dim = 2
    debug = True

    #################################################
    # Configure GMM target distribution
    #################################################
    # mean1 = np.array([0.25, 0.25])
    # cov1 = np.eye(2) * 0.01
    # mean2 = np.array([0.75, 0.75])
    # cov2 = np.eye(2) * 0.01
    # tgt = GMM(
    #     [mean1, mean2],
    #     [cov1, cov2],
    #     [0.5, 0.5]
    # )

    mean1 = np.array([0.35, 0.38])
    mean2 = np.array([0.68, 0.25])
    mean3 = np.array([0.56, 0.64])
    means = [mean1, mean2, mean3]
    cov1 = np.array([
        [0.01, 0.004],
        [0.004, 0.01]
    ])
    cov2 = np.array([
        [0.005, -0.003],
        [-0.003, 0.005]
    ])
    cov3 = np.array([
        [0.008, 0.0],
        [0.0, 0.004]
    ])
    covs = [cov1, cov2, cov3]
    tgt = GMM(means, covs, np.array([0.5,0.2,0.3]))


    s0 = np.array([0.1, 0.9])
    if debug:
        print(f'test tgt({s0}): {tgt.pdf(s0)}')
        print(f'test tgt_dx({s0}): {tgt.dpdf(s0)}')

    #################################################
    # Configure kernel
    #################################################
    kernel = Kernel(bw = 520.0, dim = 2)
    x1 = np.array([0.1, 0.3])
    x2 = np.array([0.2, 0.2])
    if debug:
        print(f'test kernel({x1}, {x2}): {kernel.eval(x1, x2)}')
        print(f'test kernel_dx1({x1}, {x2}): {kernel.grad_x1(x1, x2)}')

    #################################################
    # Configure barrier function
    #################################################
    barr = Barr(x_min=0.05, x_max=0.95)
    if debug:
        print(f'test barr({x1}): {barr.eval(x1)}')

    #################################################
    # Configure agent
    #################################################
    agent = Agent(dt = dt, dim = dim)
    if debug:    
        print(f'test agent.dyn({x1}, {x2}): {agent.dyn(x1, x2)}')

    #################################################
    # Configure controller
    #################################################
    tsteps = 100
    Q = np.eye(dim) * 1.0
    R = np.eye(dim) * 5.0
    ilqr = KES_iLQR(
        tsteps = tsteps, dt = dt, dim = dim,
        agent = agent, tgt = tgt, kernel = kernel, barr = barr,
        Q = Q, R = R, ucost_w = 0.0, barr_w = 0.0
    )

    #################################################
    # Visualization
    #################################################
    grids_x, grids_y = np.meshgrid(*[np.linspace(0.0, 1.0, 100) for _ in range(dim)])
    grids = np.array([grids_x.ravel(), grids_y.ravel()]).T 
    pdf_grids = tgt.pdf(grids).reshape((100, 100))

    # s0 = np.random.uniform(low=0.2, high=0.8, size=(dim,))
    s0 = np.array([0.48, 0.52])
    print('s0: ', s0)
    # u_traj = np.zeros((tsteps, dim)) + 1e-06
    # u_traj = np.random.uniform(low=-0.2, high=0.2, size=(tsteps,2))
    # s0 = np.array([0.1, 0.9])
    # u_traj = np.tile(np.array([0.001, 0.001]), reps=(tsteps,1))

    init_x_traj = np.array([
        np.linspace(0.0, 0.3, tsteps+1) * np.cos(np.linspace(0.0, 2*np.pi, tsteps+1)),
        np.linspace(0.0, 0.3, tsteps+1) * np.sin(np.linspace(0.0, 2*np.pi, tsteps+1))
    ]).T
    u_traj = (init_x_traj[1:, :] - init_x_traj[:-1, :]) / dt

    fig, axes = plt.subplots(1, 2, figsize=(16,6))

    ilqr.set_info_w(1.0)
    for iter in range(100):
        # if iter < 100:
        #     ilqr.set_info_w(0.0)
        # else:
        #     ilqr.set_info_w(1.0)
        s_traj = ilqr.agent.traj_sim(s0, u_traj)
        # print('s_traj:\n', s_traj)
        # print('u_traj:\n', u_traj)
        v_traj = ilqr.get_v_traj(s0, u_traj)
        step, opt_s_traj, opt_loss = ilqr.line_search(s0, u_traj, v_traj)
        u_traj += step * v_traj

        print('loss: ', opt_loss)
        print('step: ', step)

        ax = axes[0]
        ax.cla()
        ax.set_title(f'Iter: {iter}')
        ax.set_aspect('equal')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.contourf(grids_x, grids_y, pdf_grids, cmap='Reds')
        ax.plot(opt_s_traj[:,0], opt_s_traj[:,1], linestyle='-', marker='o', color='k', markersize=3, alpha=0.5)
        ax.plot([s0[0], opt_s_traj[0,0]], [s0[1], opt_s_traj[0,1]], linestyle='-', color='k', alpha=0.5)
        ax.plot(s0[0], s0[1], marker='o', color='C0', markersize=10)

        ax = axes[1]
        ax.cla()
        ax.set_aspect('auto')
        ax.set_ylim(-1, 1)
        ax.plot(u_traj)

        plt.pause(0.01)

    # np.savetxt('s_traj.txt', opt_s_traj)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()