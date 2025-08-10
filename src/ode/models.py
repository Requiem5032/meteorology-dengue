import torch
from torchdiffeq import odeint
from src.ode.math import torch_interp

AQUATIC_STATE = ['E', 'L', 'P']


def get_solution(t_eval, t_original, y0, temperature_arr, rainfall_arr, param_dict):
    solution = odeint(
        func=lambda t, y: dengue_ode_system(
            t,
            y,
            t_original,
            temperature_arr,
            rainfall_arr,
            param_dict,
        ),
        y0=y0,
        t=t_eval,
        method='dopri5',
        options={'min_step': 0.01}
    )

    return solution


def dengue_ode_system(t, y, t_original, temperature_arr, rainfall_arr, param_dict):
    current_temperature = torch_interp(t, t_original, temperature_arr)
    current_rainfall = torch_interp(t, t_original, rainfall_arr)
    meteorology_vars_dict = compute_meteorology_vars(current_temperature, current_rainfall, param_dict)
    _, E, L, P, M_s, M_e, M_i, H_s, H_e, H_i, H_r = y

    C = param_dict['C']
    sigma = param_dict['sigma']
    beta_M = param_dict['beta_M']
    theta_M = param_dict['theta_M']
    gamma = param_dict['gamma']
    mu_H = param_dict['mu_H']
    beta_H = param_dict['beta_H']
    theta_H = param_dict['theta_H']

    M = M_s + M_e + M_i
    H = H_s + H_e + H_i + H_r
    H = torch.clamp(H, min=1)
    M = torch.clamp(M, min=1)
    H_i_frac = H_i / H
    M_i_frac = M_i / M

    dHit_dt = theta_H * H_e
    dE_dt = meteorology_vars_dict['b'] * (1 - E / C) * M - (meteorology_vars_dict['F_E'] + meteorology_vars_dict['mu_E']) * E
    dL_dt = meteorology_vars_dict['F_E'] * E - (meteorology_vars_dict['F_L'] + meteorology_vars_dict['mu_L']) * L
    dP_dt = meteorology_vars_dict['F_L'] * L - (meteorology_vars_dict['F_P'] + meteorology_vars_dict['mu_P']) * P
    dMs_dt = sigma * meteorology_vars_dict['F_P'] * P - beta_M * H_i_frac * M_s - meteorology_vars_dict['mu_M'] * M_s
    dMe_dt = beta_M * H_i_frac * M_s - (theta_M + meteorology_vars_dict['mu_M']) * M_e
    dMi_dt = theta_M * M_e - meteorology_vars_dict['mu_M'] * M_i
    dHs_dt = mu_H * H - beta_H * M_i_frac * H_s - mu_H * H_s
    dHe_dt = beta_H * M_i_frac * H_s - (theta_H + mu_H) * H_e
    dHi_dt = theta_H * H_e - (gamma + mu_H) * H_i
    dHr_dt = gamma * H_i - mu_H * H_r

    # Ensure all d_dt are tensors of the same shape
    dHit_dt = torch.atleast_1d(dHit_dt)
    dE_dt = torch.atleast_1d(dE_dt)
    dL_dt = torch.atleast_1d(dL_dt)
    dP_dt = torch.atleast_1d(dP_dt)
    dMs_dt = torch.atleast_1d(dMs_dt)
    dMe_dt = torch.atleast_1d(dMe_dt)
    dMi_dt = torch.atleast_1d(dMi_dt)
    dHs_dt = torch.atleast_1d(dHs_dt)
    dHe_dt = torch.atleast_1d(dHe_dt)
    dHi_dt = torch.atleast_1d(dHi_dt)
    dHr_dt = torch.atleast_1d(dHr_dt)

    dy_dt = torch.stack([
        dHit_dt.squeeze(),
        dE_dt.squeeze(),
        dL_dt.squeeze(),
        dP_dt.squeeze(),
        dMs_dt.squeeze(),
        dMe_dt.squeeze(),
        dMi_dt.squeeze(),
        dHs_dt.squeeze(),
        dHe_dt.squeeze(),
        dHi_dt.squeeze(),
        dHr_dt.squeeze(),
    ])

    dy_dt = torch.clamp(dy_dt, min=0.0, max=1e8)
    return dy_dt


def compute_meteorology_vars(temperature, rainfall, param_dict):
    meteorology_vars_dict = {}
    temperature_funcs_dict = compute_temperature_funcs(temperature, param_dict)
    rainfall_funcs_dict = compute_rainfall_funcs(rainfall, param_dict)
    meteorology_vars_dict['mu_M'] = temperature_funcs_dict['mu_M']
    meteorology_vars_dict['b'] = param_dict['alpha_b'] * temperature_funcs_dict['u_b'] * rainfall_funcs_dict['v_b']

    for j in AQUATIC_STATE:
        meteorology_vars_dict[f'F_{j}'] = param_dict[f'alpha_{j}'] * temperature_funcs_dict[f'g_{j}'] * rainfall_funcs_dict[f'h_{j}']
        meteorology_vars_dict[f'mu_{j}'] = temperature_funcs_dict[f'p_{j}'] * rainfall_funcs_dict[f'q_{j}']

    for k, v in meteorology_vars_dict.items():
        meteorology_vars_dict[k] = torch.clamp(v, min=0.0, max=1e8)

    return meteorology_vars_dict


def compute_temperature_funcs(temperature, param_dict):
    temperature_funcs_dict = {}

    def u_b():
        a_b = param_dict['a_b']
        T_b_max = param_dict['T_b_max']
        return torch.exp(-a_b * (temperature - T_b_max) ** 2)

    def mu_M():
        c_M = param_dict['c_M']
        T_M_min = param_dict['T_M_min']
        d_M = param_dict['d_M']
        return c_M * (temperature - T_M_min) ** 2 + d_M

    def g_j(j):
        a_j = param_dict[f'a_{j}']
        T_j_max = param_dict[f'T_{j}_max']
        return torch.exp(-a_j * (temperature - T_j_max) ** 2)

    def p_j(j):
        c_j = param_dict[f'c_{j}']
        T_j_min = param_dict[f'T_{j}_min']
        d_j = param_dict[f'd_{j}']
        return c_j * (temperature - T_j_min) ** 2 + d_j

    temperature_funcs_dict['u_b'] = u_b()
    temperature_funcs_dict['mu_M'] = mu_M()

    for j in AQUATIC_STATE:
        temperature_funcs_dict[f'g_{j}'] = g_j(j)
        temperature_funcs_dict[f'p_{j}'] = p_j(j)

    for k, v in temperature_funcs_dict.items():
        temperature_funcs_dict[k] = torch.clamp(v, min=0.0, max=1e8)

    return temperature_funcs_dict


def compute_rainfall_funcs(rainfall, param_dict):
    rainfall_funcs_dict = {}

    def v_b():
        s_b = param_dict['s_b']
        r_b = param_dict['r_b']
        R_b = param_dict['R_b']
        num = (1 + s_b) * torch.exp(-r_b * (rainfall - R_b) ** 2)
        denom = torch.exp(-r_b * (rainfall - R_b) ** 2) + s_b
        return num / denom

    def h_j(j):
        s_j = param_dict[f's_{j}']
        r_j = param_dict[f'r_{j}']
        R_j = param_dict[f'R_{j}']
        num = (1 + s_j) * torch.exp(-r_j * (rainfall - R_j) ** 2)
        denom = torch.exp(-r_j * (rainfall - R_j) ** 2) + s_j
        return num / denom

    def q_j(j):
        e_j = param_dict[f'e_{j}']
        return 1 + (e_j * rainfall) / (1 + rainfall)

    rainfall_funcs_dict['v_b'] = v_b()

    for j in AQUATIC_STATE:
        rainfall_funcs_dict[f'h_{j}'] = h_j(j)
        rainfall_funcs_dict[f'q_{j}'] = q_j(j)

    for k, v in rainfall_funcs_dict.items():
        rainfall_funcs_dict[k] = torch.clamp(v, min=0.0, max=1e8)

    return rainfall_funcs_dict
