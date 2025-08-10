import yaml
import torch
import pandas as pd


def extract_params(yaml_path):
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)

    to_learn = {}
    true_parameter = {}

    for item in params:
        for key, value in item.items():
            if isinstance(value, list):
                if len(value) >= 2:
                    to_learn[key] = value
            elif value is None:
                to_learn[key] = None
            elif value is not None:
                true_parameter[key] = value

    return to_learn, true_parameter


def extract_cumulative_cases(csv_path):
    df = pd.read_csv(csv_path)
    cases = torch.tensor(df['cases'].values, dtype=torch.float32).unsqueeze(-1)
    return cases


def extract_temperature_rainfall(csv_path):
    df = pd.read_csv(csv_path)
    temperature = torch.tensor(
        df['tempC'].values, dtype=torch.float32).unsqueeze(-1)
    rainfall = torch.tensor(df['precipMM'].values,
                            dtype=torch.float32).unsqueeze(-1)
    return temperature, rainfall
