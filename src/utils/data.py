import copy
import yaml
import torch
import pandas as pd


def extract_params(yaml_path):
    with open(yaml_path, 'r') as f:
        params_dict = yaml.safe_load(f)
    params_dict = convert_to_tensor(params_dict)
    return params_dict


def get_learnable_params(params_dict) -> list[str]:
    learnable_params = []
    for key, value in params_dict.items():
        if value is None:
            learnable_params.append(key)
    return learnable_params


def get_fixed_params(params_dict) -> dict[str, torch.Tensor]:
    fixed_params = {}
    for key, value in params_dict.items():
        if value is not None:
            fixed_params[key] = value
    return fixed_params


def convert_to_tensor(param_dict, device='cpu') -> dict[str, torch.Tensor]:
    tensor_dict = copy.deepcopy(param_dict)
    for key, value in param_dict.items():
        if value is not None:
            tensor_dict[key] = torch.tensor(value, dtype=torch.float32).to(device)
    return tensor_dict


def extract_cumulative_cases(csv_path):
    df = pd.read_csv(csv_path)
    cases = torch.tensor(df['cases'].values, dtype=torch.float32).unsqueeze(-1)
    return cases


def extract_temperature_rainfall(csv_path):
    df = pd.read_csv(csv_path)
    temperature = torch.tensor(
        df['temperature_2m_mean'].values, dtype=torch.float32).unsqueeze(-1)
    rainfall = torch.tensor(df['rain_sum'].values,
                            dtype=torch.float32).unsqueeze(-1)
    return temperature, rainfall
