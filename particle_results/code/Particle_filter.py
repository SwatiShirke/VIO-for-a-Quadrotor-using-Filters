import numpy as np


class Particle_filter:
    def __init__(self, map_bounds,num_particles, acc_noise, gyro_noise, obs_cov):
        self.num_particles = num_particles
        self.map_bounds = map_bounds
        self.acc_noise = acc_noise
        self.gyro_noise = gyro_noise
        self.obs_cov = obs_cov
        self.particles = np.zeros((self.num_particles, 6))

    def init_particles_uniform(self):
        x_rng, y_rng, z_rng = self.map_bounds[0:2], self.map_bounds[2:4], self.map_bounds[4:6]
        angle_rng = (-np.pi/2, np.pi/2)
        low = np.array([x_rng[0], y_rng[0], z_rng[0], angle_rng[0], angle_rng[0], angle_rng[0]])
        high = np.array([x_rng[1], y_rng[1], z_rng[1], angle_rng[1], angle_rng[1], angle_rng[1]])

        base = np.random.uniform(low=low, high=high, size=(self.num_particles, 6))
        return np.expand_dims(np.hstack([base, np.zeros((self.num_particles, 9))]), axis=-1)

    def init_particles_gaussian(self, mean, cov):
        base = np.random.multivariate_normal(mean, cov, self.num_particles)
        return np.expand_dims(base, axis=-1)

    def compute_derivative(self, state, acc, gyro):
        dx = np.zeros((self.num_particles, 15, 1))
        phi, theta, psi = state[:, 3], state[:, 4], state[:, 5]
        R = self._rotation_matrix(phi, theta, psi)
        G = self._transformation_matrix(phi, theta)
        dx[:, 0:3] = state[:, 6:9]
        dx[:, 3:6] = np.linalg.inv(G) @ (gyro - state[:, 9:12])
        dx[:, 6:9] = np.array([0, 0, -9.81]).reshape((3,1)) + R @ (acc - state[:, 12:15])
        return dx

    def predict(self, particles, imu, dt):
        noise = np.zeros((self.num_particles, 6, 1))
        noise[:, 0:3] = np.random.normal(scale=self.gyro_noise, size=(self.num_particles, 3, 1))
        noise[:, 3:6] = np.random.normal(scale=self.acc_noise, size=(self.num_particles, 3, 1))
        gyro = np.tile(imu[:3].reshape(3, 1), (self.num_particles, 1, 1))
        acc = np.tile(imu[3:6].reshape(3, 1), (self.num_particles, 1, 1)) + noise[:, 3:6]
        dx = self.compute_derivative(particles, acc, gyro)
        dx[:, 3:6] += noise[:, :3]
        return particles + dx * dt

    def update(self, particles, measurement):
        H = np.zeros((6, 15))
        H[0:6, 0:6] = np.eye(6)
        z_pred = (H @ particles).reshape((self.num_particles, 6)) + np.diag(self.obs_cov)
        z_full = np.hstack([z_pred, np.zeros((self.num_particles, 9))])
        return self.update_weights(z_full, measurement)

    def update_weights(self, predicted, observed):
        err = predicted[:, 0:6] - np.tile(observed[0:6].flatten(), (predicted.shape[0], 1))
        weights = np.exp(-0.5 * np.sum(err ** 2, axis=1))
        total = np.sum(weights)
        if total == 0:
            raise ValueError("All weights zero.")
        return weights / total

    def resample(self, particles, weights):
        weights /= np.sum(weights)
        resampled = np.zeros((self.num_particles, 15, 1))
        r = np.random.uniform(0, 1 / self.num_particles)
        i, c = 0, weights[0]
        for k in range(self.num_particles):
            u = r + k / self.num_particles
            while u > c:
                i += 1
                c += weights[i]
            resampled[k] = particles[i]
        return resampled

    def estimate_pose(self, particles, weights, method='weighted_avg'):
        if method == 'weighted_avg':
            return np.sum(particles * weights.reshape(self.num_particles, 1, 1), axis=0)
        elif method == 'highest_weight':
            return particles[np.argmax(weights)]
        elif method == 'average':
            return np.mean(particles, axis=0)
        else:
            raise ValueError(f"Invalid method: {method}")

    @staticmethod
    def _rotation_matrix(phi, theta, psi):
        R = np.zeros((phi.shape[0], 3, 3, 1))
        R[:, 0, 0] = np.cos(psi) * np.cos(theta) - np.sin(psi) * np.sin(theta) * np.sin(phi)
        R[:, 0, 1] = -np.cos(phi) * np.sin(psi)
        R[:, 0, 2] = np.cos(psi) * np.sin(theta) + np.cos(theta) * np.sin(phi) * np.sin(psi)
        R[:, 1, 0] = np.cos(theta) * np.sin(psi) + np.cos(psi) * np.sin(phi) * np.sin(theta)
        R[:, 1, 1] = np.cos(psi) * np.cos(phi)
        R[:, 1, 2] = np.sin(psi) * np.sin(theta) - np.cos(psi) * np.cos(theta) * np.sin(phi)
        R[:, 2, 0] = -np.cos(phi) * np.sin(theta)
        R[:, 2, 1] = np.sin(phi)
        R[:, 2, 2] = np.cos(phi) * np.cos(theta)
        return R.reshape(-1, 3,3)

    @staticmethod
    def _transformation_matrix(phi, theta):
        G = np.zeros((phi.shape[0], 3, 3,1))
        G[:, 0, 0] = np.cos(theta)
        G[:, 0, 2] = -np.cos(phi) * np.sin(theta)
        G[:, 1, 1] = 1
        G[:, 1, 2] = np.sin(phi)
        G[:, 2, 0] = np.sin(theta)
        G[:, 2, 2] = np.cos(phi) * np.cos(theta)
        return G.reshape(-1, 3,3)
