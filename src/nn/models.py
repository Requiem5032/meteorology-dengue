import copy
import yaml
import random
import warnings
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tqdm.notebook import tqdm
from .ode import get_solution
from src.utils import *


class DengueNN():
    def __init__(
        self,
        device,
        data_csv_path,
        params_yaml_path,
        lr,
        epochs,
        hidden_dim,
        hidden_num,
    ):
        self.device = device
        self.param_dict = extract_params(params_yaml_path)
        self.learnable_params = get_learnable_params(self.param_dict)
        self.cumulative_cases = extract_cumulative_cases(
            data_csv_path).to(self.device)
        self.temperature_data, self.rainfall_data = extract_temperature_rainfall(
            data_csv_path)
        self.param_df = pd.DataFrame(columns=self.param_dict.keys())

        self.cumulative_cases = self.cumulative_cases.log1p()

        input_dim = len(self.cumulative_cases[0])
        output_dim = len(self.learnable_params)
        self.lr = lr
        self.epochs = epochs

        self.model = NeuralNetwork(
            input_dim,
            output_dim,
            hidden_dim,
            hidden_num,
        ).to(self.device)
        self.model.apply(init_weight)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            epochs=self.epochs,
            steps_per_epoch=1,
        )


class NeuralNetwork(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim,
            hidden_num,
    ):
        super().__init__()
        hidden_activation = nn.LeakyReLU()
        final_activation = Absolute()

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
        hidden_layers.append(nn.LayerNorm(output_dim))
        hidden_layers.append(final_activation)

        self.linears = nn.ModuleList(hidden_layers)

    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
        return x


class Absolute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(
            m.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0)


def train(sema, dengue_model, seed, result_dir):
    with sema:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            dengue_nn = copy.deepcopy(dengue_model)
            dengue_nn.model.train()
            data = dengue_nn.cumulative_cases[0]

            loss_history = []
            best_loss = float('inf')
            best_epoch = None
            best_solution = None
            best_param_dict = {}

            # Training loop
            for epoch in range(dengue_nn.epochs):
                dengue_nn.optimizer.zero_grad()
                outputs = dengue_nn.model(data)

                t_original = torch.linspace(
                    0,
                    1,
                    steps=len(dengue_nn.cumulative_cases),
                    dtype=torch.float32,
                    device=dengue_nn.device,
                )
                t_eval = torch.linspace(
                    0,
                    1,
                    steps=len(dengue_nn.cumulative_cases),
                    dtype=torch.float32,
                    device=dengue_nn.device,
                )

                for key, val in zip(dengue_nn.learnable_params, outputs):
                    dengue_nn.param_dict[key] = val

                param_list = []
                for key in dengue_nn.param_dict.keys():
                    param_list.append(
                        float(dengue_nn.param_dict[key].detach().clone().numpy()))
                dengue_nn.param_df.loc[len(dengue_nn.param_df)] = param_list

                y0_list = []
                y0_list.append(data)
                for state in outputs[:10]:
                    y0_list.append(torch.atleast_1d(state))
                y0 = torch.stack(y0_list).squeeze(-1)

                solution = get_solution(
                    t_eval=t_eval,
                    t_original=t_original,
                    y0=y0,
                    temperature_arr=dengue_nn.temperature_data,
                    rainfall_arr=dengue_nn.rainfall_data,
                    param_dict=dengue_nn.param_dict,
                )

                predicted_solution = solution.t()[0][1:]
                true_solution = dengue_nn.cumulative_cases[1:].squeeze()

                loss = dengue_nn.criterion(predicted_solution, true_solution)

                loss.backward()
                dengue_nn.optimizer.step()
                dengue_nn.scheduler.step()
                loss_history.append(loss.item())

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_solution = solution.t()[0].detach().cpu().numpy()
                    best_epoch = epoch+1
                    for key, value in dengue_nn.param_dict.items():
                        best_param_dict[key] = float(
                            value.detach().clone().numpy())

            # Creating figures
            true_data = dengue_nn.cumulative_cases
            pred_data = best_solution

            true_data_tensor_normalized = torch.tensor(
                true_data, dtype=torch.float32)
            pred_data_tensor_normalized = torch.tensor(
                pred_data, dtype=torch.float32).unsqueeze(-1)

            true_data_tensor = true_data_tensor_normalized.expm1()
            pred_data_tensor = pred_data_tensor_normalized.expm1()

            loss_normalized = dengue_nn.criterion(
                pred_data_tensor_normalized, true_data_tensor_normalized)
            loss_normalized = float(loss_normalized.detach().cpu().numpy())

            loss_unnormalized = dengue_nn.criterion(
                pred_data_tensor, true_data_tensor)
            loss_unnormalized = float(loss_unnormalized.detach().cpu().numpy())

            fig1, ax1_fig1 = plt.subplots()
            ax1_fig1.plot(true_data_tensor_normalized, label='True Cases')
            ax1_fig1.plot(pred_data_tensor_normalized, label='Predicted Cases')
            ax1_fig1.set_xlabel('Time (weeks)')
            ax1_fig1.set_ylabel('Normalized Cases')
            ax1_fig1.set_title('Calibration: True vs Predicted Dengue Cases')
            ax1_fig1.text(0.5, -0.2, f'MSE Loss: {loss_normalized:.4f}',
                          transform=ax1_fig1.transAxes, ha='center')
            ax1_fig1.legend()
            ax1_fig1.grid(True)

            fig2, ax1_fig2 = plt.subplots()
            ax1_fig2.plot(true_data_tensor, label='True Cases')
            ax1_fig2.plot(pred_data_tensor, label='Predicted Cases')
            ax1_fig2.set_xlabel('Time (weeks)')
            ax1_fig2.set_ylabel('Cumulative Cases')
            ax1_fig2.set_title('Calibration: True vs Predicted Dengue Cases')
            ax1_fig2.text(
                0.5, -0.2, f'MSE Loss: {loss:.4f}', transform=ax1_fig2.transAxes, ha='center')
            ax1_fig2.legend()
            ax1_fig2.grid(True)

            fig3, ax1_fig3 = plt.subplots()
            ax1_fig3.plot(loss_history, label='Training Loss')
            ax1_fig3.set_xlabel('Epoch')
            ax1_fig3.set_ylabel('Loss')
            ax1_fig3.set_title('Training Loss History')
            ax1_fig3.legend()
            ax1_fig3.grid(True)

            # Save results
            with open(f'{result_dir}/best_params.yaml', 'w') as f:
                yaml.safe_dump(best_param_dict, f, sort_keys=False)

            dengue_nn.param_df.to_csv(
                f'{result_dir}/param_history.csv', index=False)

            fig1.savefig(
                f'{result_dir}/calibration_normalized_cases.png',
                bbox_inches='tight',
            )
            fig2.savefig(
                f'{result_dir}/calibration_cumulative_cases.png',
                bbox_inches='tight',
            )
            fig3.savefig(
                f'{result_dir}/calibration_loss.png',
                bbox_inches='tight',
            )

            print(f'Best loss: {best_loss:.4f} at epoch {best_epoch}', flush=True)
