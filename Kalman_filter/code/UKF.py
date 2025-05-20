import numpy as np
from scipy.linalg import sqrtm
from math import sin, cos

class UKF:
    def __init__(self, cov_init=None, noise_proc=None, noise_meas=None, timestep=0.01):
        self.state_dim = 15
        self.P = cov_init if cov_init is not None else np.eye(self.state_dim)
        self.Q = noise_proc if noise_proc is not None else 0.001 * np.eye(6)
        self.R = noise_meas if noise_meas is not None else np.eye(6)
        self.dt = timestep

        # UKF parameters
        self.alpha = 0.01
        self.kappa = 1
        self.beta = 2

        # Safety limits
        self.angle_limit = 0.5
        self.vel_limit = 5.0
        self.ang_vel_limit = 2.0
        self.acc_limit = 10.0

    def get_sigma_points(self, mean, cov):
        dim = self.state_dim
        sigma = np.zeros((2 * dim + 1, dim))
        sigma[0] = mean
        lambd = self.alpha ** 2 * (dim + self.kappa) - dim
        try:
            sqrt_term = sqrtm((dim + lambd) * cov)
        except:
            sqrt_term = sqrtm((dim + lambd) * (cov + 1e-8 * np.eye(dim)))
        for i in range(dim):
            sigma[i + 1] = mean + sqrt_term[i]
            sigma[dim + i + 1] = mean - sqrt_term[i]
        return sigma.T

    def get_weights(self, dim):
        lambd = self.alpha ** 2 * (dim + self.kappa) - dim
        wm = np.full(2 * dim + 1, 1 / (2 * (dim + lambd)))
        wc = np.copy(wm)
        wm[0] = lambd / (dim + lambd)
        wc[0] = wm[0] + (1 - self.alpha ** 2 + self.beta)
        return wm, wc

    def get_R_mat(self, q):
        phi, theta, psi = q 
        R_mat = np.array([
            [cos(psi)*cos(theta)-sin(psi)*sin(theta)*sin(phi), -cos(phi)*sin(psi), cos(psi)*sin(theta)+cos(theta)*sin(phi)*sin(psi)],
            [cos(theta)*sin(psi)+cos(psi)*sin(phi)*sin(theta), cos(psi)*cos(phi), sin(psi)*sin(theta)-cos(psi)*cos(theta)*sin(phi)],
            [-cos(phi)*sin(theta), sin(phi), cos(phi)*cos(theta)]
        ])
        return R_mat


    def get_G_mat(self, q):
        phi, theta, psi = q
        G_mat = np.zeros((3, 3))
        G_mat[0, 0] = cos(theta)
        G_mat[0, 2] = -cos(phi)*sin(theta)
        G_mat[1, 1] = 1
        G_mat[1, 2] = sin(phi)
        G_mat[2, 0] = sin(theta)
        G_mat[2, 2] = cos(phi)*cos(theta)
        return G_mat
    
    def cal_dt(self, state, u):
             
        q = state[3:6]
        R_mat = self.get_R_mat(q)       
        G_mat = self.get_G_mat(q)
        
        x_dt = np.zeros(len(state))

        vel = state[6:9]
        omg = u[0:3]
        accel = u[3:6]


        x_dt[0:3] = vel
        try:
            x_dt[3:6] = np.linalg.inv(G_mat) @ (omg - state[9:12])
        except:
            x_dt[3:6] = np.linalg.inv(G_mat + 1e-6 * np.eye(3)) @ (omg - state[9:12])

        x_dt[3:6] = np.clip(x_dt[3:6], -self.ang_vel_limit, self.ang_vel_limit)
        x_dt[6:9] = np.array([0, 0, -9.81]) + R_mat @ (accel - state[12:15])
        x_dt[6:9] = np.clip(x_dt[6:9], -self.acc_limit, self.acc_limit)

        return x_dt

    def measure(self, state):
        C = np.zeros((6, self.state_dim))
        C[:6, :6] = np.eye(6)
        return C @ state

    def predict(self, mean, u, cov, Q_dyn, dt):
        sp = self.get_sigma_points(mean, cov)
        wm, wc = self.get_weights(self.state_dim)

        for i in range(sp.shape[1]):
            sp[:, i] += self.cal_dt(sp[:, i], u) * dt
            sp[6:9, i] = np.clip(sp[6:9, i], -self.vel_limit, self.vel_limit)
            if i > 0:
                ang_delta = np.mod(sp[3:6, i] - sp[3:6, 0] + np.pi, 2 * np.pi) - np.pi
                ang_delta = np.clip(ang_delta, -self.angle_limit, self.angle_limit)
                sp[3:6, i] = sp[3:6, 0] + ang_delta

        mean_pred = sp @ wm
        d = sp - mean_pred[:, None]
        cov_pred = d @ np.diag(wc) @ d.T + Q_dyn
        cov_pred = (cov_pred + cov_pred.T) / 2
        mean_pred[3:6] = np.mod(mean_pred[3:6] + np.pi, 2 * np.pi) - np.pi

        return mean_pred, cov_pred

    def update(self, mean, measurement, cov, R_meas):
        sp = self.get_sigma_points(mean, cov)
        wm, wc = self.get_weights(self.state_dim)

        z_sigma = np.array([self.measure(sp[:, i]) for i in range(sp.shape[1])]).T
        z_mean = z_sigma @ wm
        dz = z_sigma - z_mean[:, None]
        S = dz @ np.diag(wc) @ dz.T + R_meas
        S = (S + S.T) / 2 + 1e-6 * np.eye(S.shape[0])

        x_dt = sp - mean[:, None]
        cross_cov = x_dt @ np.diag(wc) @ dz.T

        
        K_gain = cross_cov @ np.linalg.pinv(S)

        mean_new = mean + K_gain @ (measurement - z_mean)
        cov_new = cov - K_gain @ S @ K_gain.T
        cov_new = (cov_new + cov_new.T) / 2

        min_eigen = np.min(np.real(np.linalg.eigvals(cov_new)))
        if min_eigen < 1e-6:
            cov_new += (1e-6 - min_eigen) * np.eye(self.state_dim)

        mean_new[3:6] = np.mod(mean_new[3:6] + np.pi, 2 * np.pi) - np.pi

        return mean_new, cov_new
