import torch
from torchdiffeq import odeint
from src.utils import *

_AQUATIC_STATE = ['E', 'L', 'P']


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
        rtol=1e-3,
        atol=1e-6,
        method='dopri5',
    )

    return solution


def dengue_ode_system(t, y, t_original, temperature_arr, rainfall_arr, param_dict):
    current_temperature = interp1d(t, t_original, temperature_arr)
    current_rainfall = interp1d(t, t_original, rainfall_arr)
    _raise_if_not_finite(current_temperature,
                         'interp1d(t, t_original, temperature_arr)', t)
    _raise_if_not_finite(
        current_rainfall, 'interp1d(t, t_original, rainfall_arr)', t)

    meteorology_vars_dict = compute_meteorology_vars(
        current_temperature, current_rainfall, param_dict, t)

    _, E, L, P, M_s, M_e, M_i, H_s, H_e, H_i, H_r = y
    _raise_if_not_finite(y, 'state y', t)

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
    _raise_if_not_finite(M, 'M_s + M_e + M_i', t)
    _raise_if_not_finite(H, 'H_s + H_e + H_i + H_r', t)
    H_i_frac = smooth_safe_divide(H_i, H)
    M_i_frac = smooth_safe_divide(M_i, M)
    E_over_C = smooth_safe_divide(E, C)
    _raise_if_not_finite(H_i_frac, 'smooth_safe_divide(H_i, H)', t)
    _raise_if_not_finite(M_i_frac, 'smooth_safe_divide(M_i, M)', t)
    _raise_if_not_finite(E_over_C, 'smooth_safe_divide(E, C)', t)

    dHit_dt = theta_H * H_e
    dE_dt = meteorology_vars_dict['b'] * (1 - E_over_C) * M - (
        meteorology_vars_dict['F_E'] + meteorology_vars_dict['mu_E']) * E
    dL_dt = meteorology_vars_dict['F_E'] * E - \
        (meteorology_vars_dict['F_L'] + meteorology_vars_dict['mu_L']) * L
    dP_dt = meteorology_vars_dict['F_L'] * L - \
        (meteorology_vars_dict['F_P'] + meteorology_vars_dict['mu_P']) * P
    dMs_dt = sigma * meteorology_vars_dict['F_P'] * P - beta_M * \
        H_i_frac * M_s - meteorology_vars_dict['mu_M'] * M_s
    dMe_dt = beta_M * H_i_frac * M_s - \
        (theta_M + meteorology_vars_dict['mu_M']) * M_e
    dMi_dt = theta_M * M_e - meteorology_vars_dict['mu_M'] * M_i
    dHs_dt = mu_H * H - beta_H * M_i_frac * H_s - mu_H * H_s
    dHe_dt = beta_H * M_i_frac * H_s - (theta_H + mu_H) * H_e
    dHi_dt = theta_H * H_e - (gamma + mu_H) * H_i
    dHr_dt = gamma * H_i - mu_H * H_r
    _raise_if_not_finite(dHit_dt, 'theta_H * H_e', t)
    _raise_if_not_finite(dE_dt, "b * (1 - E / C) * M - (F_E + mu_E) * E", t)
    _raise_if_not_finite(dL_dt, 'F_E * E - (F_L + mu_L) * L', t)
    _raise_if_not_finite(dP_dt, 'F_L * L - (F_P + mu_P) * P', t)
    _raise_if_not_finite(
        dMs_dt, 'sigma * F_P * P - beta_M * H_i_frac * M_s - mu_M * M_s', t)
    _raise_if_not_finite(
        dMe_dt, 'beta_M * H_i_frac * M_s - (theta_M + mu_M) * M_e', t)
    _raise_if_not_finite(dMi_dt, 'theta_M * M_e - mu_M * M_i', t)
    _raise_if_not_finite(
        dHs_dt, 'mu_H * H - beta_H * M_i_frac * H_s - mu_H * H_s', t)
    _raise_if_not_finite(
        dHe_dt, 'beta_H * M_i_frac * H_s - (theta_H + mu_H) * H_e', t)
    _raise_if_not_finite(dHi_dt, 'theta_H * H_e - (gamma + mu_H) * H_i', t)
    _raise_if_not_finite(dHr_dt, 'gamma * H_i - mu_H * H_r', t)

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
    _raise_if_not_finite(dy_dt, 'stacked dy_dt', t)

    return dy_dt


def compute_meteorology_vars(temperature, rainfall, param_dict, t=None):
    meteorology_vars_dict = {}
    temperature_funcs_dict = compute_temperature_funcs(
        temperature, param_dict, t)
    rainfall_funcs_dict = compute_rainfall_funcs(rainfall, param_dict, t)
    meteorology_vars_dict['mu_M'] = temperature_funcs_dict['mu_M']
    meteorology_vars_dict['b'] = param_dict['alpha_b'] * \
        temperature_funcs_dict['u_b'] * rainfall_funcs_dict['v_b']
    _raise_if_not_finite(
        meteorology_vars_dict['b'],
        'alpha_b * u_b * v_b',
        t,
    )

    for j in _AQUATIC_STATE:
        meteorology_vars_dict[f'F_{j}'] = param_dict[f'alpha_{j}'] * \
            temperature_funcs_dict[f'g_{j}'] * rainfall_funcs_dict[f'h_{j}']
        meteorology_vars_dict[f'mu_{j}'] = temperature_funcs_dict[f'p_{j}'] * \
            rainfall_funcs_dict[f'q_{j}']
        _raise_if_not_finite(
            meteorology_vars_dict[f'F_{j}'],
            f'alpha_{j} * g_{j} * h_{j}',
            t,
        )
        _raise_if_not_finite(
            meteorology_vars_dict[f'mu_{j}'],
            f'p_{j} * q_{j}',
            t,
        )

    return meteorology_vars_dict


def compute_temperature_funcs(temperature, param_dict, t=None):
    temperature_funcs_dict = {}

    def u_b():
        a_b = param_dict['a_b']
        T_b_max = param_dict['T_b_max']
        u_b_val = torch.exp(-a_b * (temperature - T_b_max) ** 2)
        _raise_if_not_finite(
            u_b_val, 'exp(-a_b * (temperature - T_b_max)^2)', t)
        return u_b_val

    def mu_M():
        c_M = param_dict['c_M']
        T_M_min = param_dict['T_M_min']
        d_M = param_dict['d_M']
        mu_m_val = c_M * (temperature - T_M_min) ** 2 + d_M
        _raise_if_not_finite(
            mu_m_val, 'c_M * (temperature - T_M_min)^2 + d_M', t)
        return mu_m_val

    def g_j(j):
        a_j = param_dict[f'a_{j}']
        T_j_max = param_dict[f'T_{j}_max']
        g_j_val = torch.exp(-a_j * (temperature - T_j_max) ** 2)
        _raise_if_not_finite(
            g_j_val, f'exp(-a_{j} * (temperature - T_{j}_max)^2)', t)
        return g_j_val

    def p_j(j):
        c_j = param_dict[f'c_{j}']
        T_j_min = param_dict[f'T_{j}_min']
        d_j = param_dict[f'd_{j}']
        p_j_val = c_j * (temperature - T_j_min) ** 2 + d_j
        _raise_if_not_finite(
            p_j_val, f'c_{j} * (temperature - T_{j}_min)^2 + d_{j}', t)
        return p_j_val

    temperature_funcs_dict['u_b'] = u_b()
    temperature_funcs_dict['mu_M'] = mu_M()

    for j in _AQUATIC_STATE:
        temperature_funcs_dict[f'g_{j}'] = g_j(j)
        temperature_funcs_dict[f'p_{j}'] = p_j(j)

    return temperature_funcs_dict


def compute_rainfall_funcs(rainfall, param_dict, t=None):
    rainfall_funcs_dict = {}

    def v_b():
        s_b = param_dict['s_b']
        r_b = param_dict['r_b']
        R_b = param_dict['R_b']
        num = (1 + s_b) * torch.exp(-r_b * (rainfall - R_b) ** 2)
        denom = torch.exp(-r_b * (rainfall - R_b) ** 2) + s_b
        _raise_if_not_finite(
            num, '(1 + s_b) * exp(-r_b * (rainfall - R_b)^2)', t)
        _raise_if_not_finite(denom, 'exp(-r_b * (rainfall - R_b)^2) + s_b', t)
        v_b_val = smooth_safe_divide(num, denom)
        _raise_if_not_finite(
            v_b_val, 'smooth_safe_divide(num, denom) in v_b', t)
        return v_b_val

    def h_j(j):
        s_j = param_dict[f's_{j}']
        r_j = param_dict[f'r_{j}']
        R_j = param_dict[f'R_{j}']
        num = (1 + s_j) * torch.exp(-r_j * (rainfall - R_j) ** 2)
        denom = torch.exp(-r_j * (rainfall - R_j) ** 2) + s_j
        _raise_if_not_finite(
            num, f'(1 + s_{j}) * exp(-r_{j} * (rainfall - R_{j})^2)', t)
        _raise_if_not_finite(
            denom, f'exp(-r_{j} * (rainfall - R_{j})^2) + s_{j}', t)
        h_j_val = smooth_safe_divide(num, denom)
        _raise_if_not_finite(
            h_j_val, f'smooth_safe_divide(num, denom) in h_{j}', t)
        return h_j_val

    def q_j(j):
        e_j = param_dict[f'e_{j}']
        q_denom = 1 + rainfall
        _raise_if_not_finite(q_denom, f'1 + rainfall denominator in q_{j}', t)
        q_j_val = 1 + smooth_safe_divide(e_j * rainfall, q_denom)
        _raise_if_not_finite(
            q_j_val, f'1 + smooth_safe_divide(e_{j} * rainfall, 1 + rainfall)', t)
        return q_j_val

    rainfall_funcs_dict['v_b'] = v_b()

    for j in _AQUATIC_STATE:
        rainfall_funcs_dict[f'h_{j}'] = h_j(j)
        rainfall_funcs_dict[f'q_{j}'] = q_j(j)

    return rainfall_funcs_dict


def _raise_if_not_finite(value, expr_name, t=None):
    value_tensor = torch.as_tensor(value)
    if not torch.isfinite(value_tensor).all():
        if t is None:
            raise FloatingPointError(
                f'Non-finite value in expression {expr_name}: {value_tensor}'
            )

        t_value = torch.as_tensor(t).detach().cpu().reshape(-1)[0].item()
        raise FloatingPointError(
            f'Non-finite value in expression {expr_name} at t={t_value}: {value_tensor}'
        )
