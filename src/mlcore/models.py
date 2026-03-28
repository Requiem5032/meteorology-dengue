import torch
import torch.nn as nn
import pandas as pd

from src.utils import (
    extract_cumulative_cases,
    extract_temperature_rainfall,
    extract_params_yaml,
    get_learnable_params,
)


class DengueNN():
    def __init__(
        self,
        device,
        data_csv_path,
        params_yaml_path,
        lr,
        betas,
        weight_decay,
        epochs,
        hidden_dim,
        hidden_num,
    ):
        self.device = device
        self.data_csv_path = data_csv_path
        self.param_dict = extract_params_yaml(params_yaml_path)
        self.learnable_params = get_learnable_params(self.param_dict)
        self.cumulative_cases = extract_cumulative_cases(
            data_csv_path).to(self.device)
        self.temperature_data, self.rainfall_data = extract_temperature_rainfall(
            data_csv_path)
        self.param_df = pd.DataFrame(columns=self.param_dict.keys())
        self.normalized_data = self.cumulative_cases.log1p()

        input_dim = len(self.cumulative_cases[0])
        output_dim = len(self.learnable_params)
        self.lr = lr
        self.epochs = epochs

        self.model = _NeuralNetwork(
            input_dim,
            output_dim,
            hidden_dim,
            hidden_num,
        ).to(self.device)
        self.model.apply(_init_weight)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            epochs=self.epochs,
            steps_per_epoch=1,
        )


class _NeuralNetwork(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim,
            hidden_num,
    ):
        super().__init__()
        hidden_activation = nn.Tanh()
        final_activation = nn.Softplus()

        hidden_layers = []
        hidden_layers.append(nn.Linear(input_dim, hidden_dim))
        hidden_layers.append(nn.LayerNorm(hidden_dim))
        hidden_layers.append(hidden_activation)

        for _ in range(hidden_num):
            hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            hidden_layers.append(nn.LayerNorm(hidden_dim))
            hidden_layers.append(hidden_activation)
            hidden_layers.append(nn.Dropout(p=0.1))

        hidden_layers.append(nn.Linear(hidden_dim, output_dim))
        hidden_layers.append(final_activation)

        self.linears = nn.ModuleList(hidden_layers)

    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
        return x


def _init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(
            m.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0)
