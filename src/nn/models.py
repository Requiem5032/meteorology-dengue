import yaml
import torch
import torch.nn as nn

from tqdm.notebook import tqdm
from src.nn.ode import get_solution
from src.utils.data import (
    extract_params,
    get_learnable_params,
    extract_cumulative_cases,
    extract_temperature_rainfall,
)


class DengueNN():
    def __init__(
        self,
        device,
        location,
        data_csv_path,
        params_yaml_path,
        lr,
        epochs,
        hidden_dim,
        hidden_num,
    ):
        self.device = device
        self.location = location
        self.param_dict = extract_params(params_yaml_path)
        self.learnable_params = get_learnable_params(self.param_dict)
        self.cumulative_cases = extract_cumulative_cases(
            data_csv_path).to(self.device)
        self.temperature_data, self.rainfall_data = extract_temperature_rainfall(
            data_csv_path)

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

    def train(self):
        self.model.train()
        data = self.cumulative_cases[0]

        progress_bar = tqdm(range(self.epochs), desc='Epochs')
        loss_history = []
        best_loss = float('inf')
        best_solution = None
        best_param_dict = {}
        for epoch in progress_bar:
            self.optimizer.zero_grad()
            outputs = self.model(data)

            # t_original = torch.arange(
            #     0,
            #     len(self.cumulative_cases),
            #     step=1,
            #     dtype=torch.float32,
            #     device=self.device,
            # )
            # t_eval = torch.arange(
            #     0,
            #     len(self.cumulative_cases),
            #     step=1,
            #     dtype=torch.float32,
            #     device=self.device,
            # )

            t_original = torch.linspace(
                0,
                1,
                steps=len(self.cumulative_cases),
                dtype=torch.float32,
                device=self.device,
            )
            t_eval = torch.linspace(
                0,
                1,
                steps=len(self.cumulative_cases),
                dtype=torch.float32,
                device=self.device,
            )

            for key, val in zip(self.learnable_params, outputs):
                self.param_dict[key] = val

            y0_list = []
            y0_list.append(data)
            for state in outputs[:10]:
                y0_list.append(torch.atleast_1d(state))
            y0 = torch.stack(y0_list).squeeze(-1)

            solution = get_solution(
                t_eval=t_eval,
                t_original=t_original,
                y0=y0,
                temperature_arr=self.temperature_data,
                rainfall_arr=self.rainfall_data,
                param_dict=self.param_dict,
            )

            predicted_solution = solution.t()[0][1:]
            true_solution = self.cumulative_cases[1:].squeeze()

            loss = self.criterion(predicted_solution, true_solution)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            progress_bar.set_description(
                f'Epoch {epoch+1}/{self.epochs} | Loss: {loss.item():.4f}')
            loss_history.append(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_solution = solution.t()[0].detach().cpu().numpy()
                for key, value in self.param_dict.items():
                    best_param_dict[key] = float(value.detach().clone().numpy())

        with open(f'results/{self.location}/best_params.yaml', 'w') as f:
            yaml.dump(best_param_dict, f)
        print(f'Best Loss: {best_loss:.4f}')
        return loss_history, best_solution


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
