import numpy as np
import scipy.io as sio
from scipy.linalg import block_diag
import time
import matplotlib.pyplot as plt
from Obs_model_Kalman import Observation_model
import UKF


def get_environment_bounds():
    return np.array([0, 3.4, 0, 2.36, 0, 2])


def run_simulation(filename, num_particles, method, Qa, Qg):
    mat_data = sio.loadmat(filename, simplify_cells=True)
    imu_data = mat_data['data']
    ground_truth = mat_data['vicon']
    gt_timestamps = mat_data['time']

    obs_model = Observation_model()
    R = obs_model.get_covar_mat()

    map_bounds = get_environment_bounds()

    ## state covar 
    covar = 0.1
    randon_bias_walk = 0.1
    P_mat = covar * np.eye(15)
    P_mat[9:15, 9:15] = randon_bias_walk *  np.eye(6)

    ##pprocess noise 
    p_noise_var = 1
    Q_ag = np.eye(6)
    Q_ag[0:3, 0:3]  = Q_ag[0:3, 0:3] * Qg
    Q_ag[3:6, 3:6] = Q_ag[3:6, 3:6] * Qa
    B_mat = np.zeros((15,6))
    B_mat[0:6, 0:6] = np.eye(6)  
    ukf = UKF.UKF(P_mat, Q_ag, R, 0.01)

    

    state_dim = 6
    num_timesteps = len(imu_data)
    est_poses = np.zeros((state_dim, num_timesteps))
    filtered_poses = np.zeros((state_dim, num_timesteps))
    particle_trace = np.zeros((num_particles, state_dim, num_timesteps))
    timestamps = np.zeros(num_timesteps)

    t_prev = 0
    i = 0
    start_time = time.time()
    state = np.zeros(15)
    for idx, data_point in enumerate(imu_data):
        meas_pose = obs_model.measure_drone_pose(data_point)
        if meas_pose is None:
            continue
        #print(meas_pose)
        if len(meas_pose) == 0:
            continue
        meas_pose = meas_pose.reshape(state_dim)
        est_poses[:, i] = meas_pose

        dt = data_point["t"] - t_prev

        #print(data_point)

        try:
            control_input = np.concatenate((data_point['drpy'], data_point['acc']))
        except: 
            control_input = np.concatenate((data_point['omg'], data_point['acc']))

        measurement = np.hstack([meas_pose[:3], meas_pose[:3], np.zeros(9)])

       
        Q = (dt * B_mat) @ Q_ag @ (dt * B_mat).T
        state, P_mat = ukf.predict(state, control_input, P_mat, Q,  dt)
        state, P_mat = ukf.update(state, meas_pose, P_mat, R)       

        filtered_poses[:, i] = state[0:6]
        t_prev = data_point["t"]
        timestamps[i] = t_prev
        i += 1

    return est_poses, timestamps, ground_truth, gt_timestamps, filtered_poses


def plot_results(est_data, est_time, gt_data, gt_time, filter_data):    
    # --- 1. 3D Trajectory Plot ---
    fig_traj = plt.figure(figsize=(8, 6))
    ax_traj = fig_traj.add_subplot(111, projection='3d')
    ax_traj.plot(gt_data[0], gt_data[1], gt_data[2], color='green', linewidth=1.5, label='Ground Truth')
    ax_traj.scatter(est_data[0], est_data[1], est_data[2], color='crimson', label='Estimated', s=4, alpha=0.6)
    ax_traj.scatter(filter_data[0], filter_data[1], filter_data[2], color='royalblue', label='Filtered', s=4, alpha=0.6)
    ax_traj.set_title('3D Trajectory', fontsize=14)
    ax_traj.set_xlabel('X', fontsize=12)
    ax_traj.set_ylabel('Y', fontsize=12)
    ax_traj.set_zlabel('Z', fontsize=12)
    ax_traj.legend()
    ax_traj.grid(True)

    # --- 2. Position vs Time ---
    fig_pos, ax_pos = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    pos_labels = ['X', 'Y', 'Z']
    colors = ['green', 'crimson', 'royalblue']
    for i in range(3):
        ax_pos[i].plot(gt_time, gt_data[i], label='GT', color=colors[0], linewidth=1.2)
        ax_pos[i].plot(est_time, est_data[i], label='Estimated', color=colors[1], linestyle='--')
        ax_pos[i].plot(est_time, filter_data[i], label='Filtered', color=colors[2], linestyle='-.')
        ax_pos[i].set_ylabel(pos_labels[i], fontsize=11)
        ax_pos[i].legend(loc='upper right')
        ax_pos[i].grid(True)
    ax_pos[0].set_title('Position vs Time', fontsize=14)
    ax_pos[2].set_xlabel('Time (s)', fontsize=12)

    # --- 3. Orientation vs Time ---
    fig_ori, ax_ori = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    angle_labels = ['Roll', 'Pitch', 'Yaw']
    gt_idx = [3, 4, 5]
    est_idx = [5, 4, 3]
    filt_idx = [3, 4, 5]
    for i in range(3):
        ax_ori[i].plot(gt_time, gt_data[gt_idx[i]], label='GT', color=colors[0], linewidth=1.2)
        ax_ori[i].plot(est_time, est_data[est_idx[i]], label='Estimated', color=colors[1], linestyle='--')
        ax_ori[i].plot(est_time, filter_data[filt_idx[i]], label='Filtered', color=colors[2], linestyle='-.')
        ax_ori[i].set_ylabel(angle_labels[i], fontsize=11)
        ax_ori[i].legend(loc='upper right')
        ax_ori[i].grid(True)
    ax_ori[0].set_title('Orientation vs Time', fontsize=14)
    ax_ori[2].set_xlabel('Time (s)', fontsize=12)

    # # --- RMSE Computation (no plot) ---
    # est_rmse, filter_rmse = compute_rmse_only(est_data, est_time, gt_data, gt_time, filter_data)
    # print(f"Mean RMSE (Estimated): {np.mean(est_rmse):.4f}")
    # print(f"Mean RMSE (Filtered):  {np.mean(filter_rmse):.4f}")

    plt.tight_layout()
    plt.show()

def compute_rmse_only(est_data, est_time, gt_data, gt_time, filter_data):
    rmse_est = np.zeros(len(est_data[0]))
    rmse_filter = np.zeros(len(filter_data[0]))
    for i in range(len(est_data[0])):
        gt_idx = np.argmin(np.abs(gt_time - est_time[i]))
        rmse_est[i] = np.sqrt(np.mean((est_data[:3, i] - gt_data[:3, gt_idx]) ** 2))
        rmse_filter[i] = np.sqrt(np.mean((filter_data[:3, i] - gt_data[:3, gt_idx]) ** 2))
    print(f"Mean RMSE (Estimated): {np.mean(rmse_est):.4f}")
    print(f"Mean RMSE (Filtered):  {np.mean(rmse_filter):.4f}")
    return rmse_est, rmse_filter


if __name__ == "__main__":
    selection_method = "weighted_avg"
    no_particles = 5000
    Qa = 1000
    Qg = 0.01
    results = run_simulation("data/studentdata1.mat", no_particles,selection_method, Qa, Qg)
    est_data, est_times, gt_data, gt_times, filt_data = results
    compute_rmse_only(est_data, est_times, gt_data, gt_times, filt_data)
    plot_results(est_data, est_times, gt_data, gt_times, filt_data)
    