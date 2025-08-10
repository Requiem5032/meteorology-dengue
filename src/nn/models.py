import torch
import torch.nn as nn

from tqdm.notebook import tqdm
from src.ode.models import get_solution
from src.utils.data import extract_params, extract_cumulative_cases, extract_temperature_rainfall


class DengueNN():
    def __init__(
        self,
        device,
        hidden_dim=64,
        hidden_num=2,
        lr=1e-4,
    ):
        self.device = device
        self.to_learn_params, self.true_params = extract_params(
            'data/params.yaml')
        self.cumulative_cases = extract_cumulative_cases(
            'data/data_bello_cumulative.csv')
        self.temperature_data, self.rainfall_data = extract_temperature_rainfall(
            'data/weather_weekly.csv')

        input_dim = len(self.cumulative_cases[0])
        output_dim = len(self.to_learn_params)

        self.model = NeuralNetwork(
            input_dim, output_dim, hidden_dim, hidden_num)
        self.model.apply(init_weight)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, epochs):
        self.model.train()

        progress_bar = tqdm(range(epochs), desc="Epochs")
        loss_history = []
        for epoch in progress_bar:
            self.optimizer.zero_grad()
            outputs = self.model(self.cumulative_cases[0])

            t_original = torch.arange(
                0, len(self.cumulative_cases), step=1, dtype=torch.float32)
            t_eval = torch.arange(
                0, len(self.cumulative_cases), step=1, dtype=torch.float32)

            def scale_output(val, bounds):
                if isinstance(bounds, list) and len(bounds) == 2:
                    min_val, max_val = bounds
                    # Scale output (assume val in [0,1] due to sigmoid)
                    return min_val + (max_val - min_val) * val
                else:
                    # No scaling if no bounds
                    return val

            to_learn_param_keys = list(self.to_learn_params.keys())
            to_learn_param_dict = {}
            for i, k in enumerate(to_learn_param_keys):
                bounds = self.to_learn_params[k]
                val = outputs[i]
                to_learn_param_dict[k] = scale_output(val, bounds)

            param_dict = {**to_learn_param_dict, **self.true_params}

            y0_list = []
            y0_list.append(self.cumulative_cases[0])
            for state in outputs[:10]:
                y0_list.append(torch.atleast_1d(state))
            y0 = torch.stack(y0_list).squeeze(-1)

            solution = get_solution(
                t_eval=t_eval,
                t_original=t_original,
                y0=y0,
                temperature_arr=self.temperature_data,
                rainfall_arr=self.rainfall_data,
                param_dict=param_dict,
            )
            loss = self.criterion(
                solution.t()[0][1:], self.cumulative_cases[1:].squeeze())

            loss.backward()
            self.optimizer.step()
            progress_bar.set_description(
                f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
            loss_history.append(loss.item())
            # Optionally, print details for debugging
            print(f'Solution: {solution.t()[0][1:]}')
            # print(f'True cases: {self.cumulative_cases[1:].squeeze()}')
            # print(f'Epoch loss: {loss.item()}')

        return loss_history


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_num):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.hidden_activation = nn.ReLU()
        self.final_activation = nn.Sigmoid()

        hidden_layers = []
        for _ in range(hidden_num):
            hidden_layers.append(self.fc2)
            hidden_layers.append(self.hidden_activation)
        self.linears = nn.ModuleList(hidden_layers)

    def forward(self, x):
        x = self.fc1(x)
        x = self.hidden_activation(x)
        for layer in self.linears:
            x = layer(x)
        x = self.fc3(x)
        x = self.final_activation(x)
        return x


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0.01)
