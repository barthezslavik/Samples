import numpy as np

T_c = 5 * 4e5
T_r = 4e5

def calculate_causal_entropic_force(cur_macrostate):
    # Monte Carlo path sampling
    dt = 0.05
    num_sample_paths = 100
    num_time_steps = 100
    cur_macrostate = np.array([np.pi, -0.2]).reshape(-1, 1)

    num_dofs = cur_macrostate.shape[0]
    sample_paths = np.zeros((num_dofs, num_time_steps, num_sample_paths))
    log_volumes = np.zeros((1, 1, num_sample_paths))
    initial_force = np.zeros((num_dofs, 1, num_sample_paths))
    sigma = [[1, 0], [0, 1]];
    R = np.linalg.cholesky(sigma)
    
    for i in range(num_sample_paths):
        cur_path = np.zeros((num_dofs, num_time_steps))
        cur_state = cur_macrostate
        print((np.zeros((num_dofs//2, 1))))
        print((np.random.randn(1, num_dofs//2) @ R).reshape(-1, 1))
        # thermal_noise = np.concatenate((np.zeros((num_dofs//2, 1)), (np.random.randn(1, num_dofs//2) @ R).reshape(-1, 1)), axis=0)
        # initial_force[:,:,i] = thermal_noise
        for n in range(num_time_steps):
            cur_path[:, n] = cur_state
            localincr = tdmethod(eof, cur_state, dt)
            cur_state = cur_state + localincr #+ thermal_noise
            cur_state[0] = np.mod(cur_state[0], 2*np.pi)
            # thermal_noise = np.concatenate((np.zeros((num_dofs//2, 1)), (np.random.randn(1, num_dofs//2) @ R).reshape(-1, 1)), axis=0)
        sample_paths[:,:,i] = cur_path
        log_volumes[:,:,i] = -np.log(get_volume(cur_path))
    
    # Kernel density estimation of log volume fractions
    log_volume_fracs = log_volumes - np.sum(log_volumes, axis=2)
    force = np.sum(np.multiply(initial_force, log_volume_fracs), axis=2)
    force = 2 * (T_c / T_r) * (1.0 / num_sample_paths) * force
    
    return force

calculate_causal_entropic_force(np.array([np.pi, -0.2]).reshape(-1, 1))