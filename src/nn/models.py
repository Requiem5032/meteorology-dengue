import copy
import yaml
import random
import warnings
import gc
import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mlflow

from tqdm import tqdm
from src.ode import *
from src.utils import *


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
        self.optimizer = torch.optim.Adam(
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


def save_figures(dengue_nn, best_solution, best_loss, loss_history, figure_dir):
    """Create and save calibration figures."""
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

    # Figure 1: Normalized cases
    fig1, ax1 = plt.subplots()
    ax1.plot(true_data_tensor_normalized.cpu().numpy(), label='True Cases')
    ax1.plot(pred_data_tensor_normalized.cpu().numpy(),
             label='Predicted Cases')
    ax1.set_xlabel('Time (weeks)')
    ax1.set_ylabel('Normalized Cases')
    ax1.set_title('Calibration: True vs Predicted Dengue Cases')
    ax1.text(0.5, -0.2, f'MSE Loss: {loss_normalized:.4f}',
             transform=ax1.transAxes, ha='center')
    ax1.legend()
    ax1.grid(True)
    fig1.savefig(
        f'{figure_dir}/calibration_normalized_cases.png',
        bbox_inches='tight',
    )
    plt.close(fig1)

    # Figure 2: Cumulative cases
    fig2, ax2 = plt.subplots()
    ax2.plot(true_data_tensor.cpu().numpy(), label='True Cases')
    ax2.plot(pred_data_tensor.cpu().numpy(), label='Predicted Cases')
    ax2.set_xlabel('Time (weeks)')
    ax2.set_ylabel('Cumulative Cases')
    ax2.set_title('Calibration: True vs Predicted Dengue Cases')
    ax2.text(0.5, -0.2, f'MSE Loss: {loss_unnormalized:.4f}',
             transform=ax2.transAxes, ha='center')
    ax2.legend()
    ax2.grid(True)
    fig2.savefig(
        f'{figure_dir}/calibration_cumulative_cases.png',
        bbox_inches='tight',
    )
    plt.close(fig2)

    # Figure 3: Loss history
    fig3, ax3 = plt.subplots()
    ax3.plot(loss_history, label='Calibration Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Calibration Loss History')
    ax3.text(0.5, -0.2, f'Best Loss: {best_loss:.4f}',
             transform=ax3.transAxes, ha='center')
    ax3.legend()
    ax3.grid(True)
    fig3.savefig(
        f'{figure_dir}/calibration_loss.png',
        bbox_inches='tight',
    )
    plt.close(fig3)

    return loss_normalized, loss_unnormalized


def train(dengue_model, seed, result_dir, figure_dir, location='unknown'):
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
        best_model_state = None

        early_stopping_patience = 20
        early_stopping_min_delta = 1e-6
        early_stopping_warmup = 10
        epochs_without_improvement = 0
        early_stopped = False

        # Training loop
        progress_bar = tqdm(range(dengue_nn.epochs), desc='Training')
        for epoch in progress_bar:
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
            progress_bar.set_description(
                f'Epoch {epoch+1}/{dengue_nn.epochs}, Loss: {loss.item():.6f}')

            if loss.item() < (best_loss - early_stopping_min_delta):
                best_loss = loss.item()
                best_solution = solution.t()[0].detach().cpu().numpy()
                best_epoch = epoch+1
                for key, value in dengue_nn.param_dict.items():
                    best_param_dict[key] = float(
                        value.detach().clone().numpy())
                best_model_state = copy.deepcopy(dengue_nn.model.state_dict())
                epochs_without_improvement = 0
            else:
                if epoch >= early_stopping_warmup:
                    epochs_without_improvement += 1

                    if epochs_without_improvement >= early_stopping_patience:
                        early_stopped = True
                        progress_bar.set_description(
                            f'Early stopping at epoch {epoch+1}/{dengue_nn.epochs}, Best Loss: {best_loss:.6f}')
                        break

        # Create and save figures after training
        loss_normalized, loss_unnormalized = save_figures(
            dengue_nn, best_solution, best_loss, loss_history, figure_dir
        )

        # Save results
        with open(f'{result_dir}/best_params.yaml', 'w') as f:
            yaml.safe_dump(best_param_dict, f, sort_keys=False)
        with open(f'{result_dir}/loss_result.yaml', 'w') as f:
            yaml.safe_dump(
                {
                    'best_loss': float(best_loss),
                    'best_epoch': int(best_epoch),
                    'final_loss': float(loss_history[-1]),
                    'normalized_loss': float(loss_normalized),
                    'unnormalized_loss': float(loss_unnormalized),
                },
                f,
                sort_keys=False,
            )

        dengue_nn.param_df.to_csv(
            f'{result_dir}/param_history.csv', index=False)
        if best_model_state is not None:
            dengue_nn.model.load_state_dict(best_model_state)
            mlflow.pytorch.log_model(
                dengue_nn.model, name=f'calibrated_dengue_nn_{location}_seed_{seed}')

        # Clean up memory
        final_loss_value = float(
            loss_history[-1]) if len(loss_history) > 0 else None
        del dengue_nn
        del data
        del best_solution
        del loss_history
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        plt.close('all')
        gc.collect()

    return {
        'result_dir': result_dir,
        'best_loss': float(best_loss),
        'best_epoch': int(best_epoch) if best_epoch is not None else None,
        'final_loss': final_loss_value,
        'normalized_loss': float(loss_normalized),
        'unnormalized_loss': float(loss_unnormalized),
        'early_stopped': early_stopped,
    }


_PROJECTION_STATE_ORDER = [
    'E_0', 'L_0', 'P_0', 'M_s_0', 'M_e_0', 'M_i_0',
    'H_s_0', 'H_e_0', 'H_i_0', 'H_r_0',
]


def run_projection(location, result_dir, figure_dir):
    """Run forward projection using calibrated params on the projection dataset."""
    data_csv_path = f'data/projection/{location}/data.csv'
    predicted_params_path = f'{result_dir}/best_params.yaml'

    if not os.path.exists(data_csv_path) or not os.path.exists(predicted_params_path):
        return None

    with torch.no_grad():
        param_dict = extract_params(predicted_params_path)
        cumulative_cases = extract_cumulative_cases(data_csv_path).log1p()
        temperature_data, rainfall_data = extract_temperature_rainfall(
            data_csv_path)

        t_original = torch.linspace(0, 1, steps=len(
            cumulative_cases), dtype=torch.float32)
        t_eval = torch.linspace(0, 1, steps=len(
            cumulative_cases), dtype=torch.float32)

        y0_list = [cumulative_cases[0]]
        for state_name in _PROJECTION_STATE_ORDER:
            y0_list.append(torch.atleast_1d(param_dict[state_name]))
        y0 = torch.stack(y0_list).squeeze(-1)

        solution = get_solution(
            t_eval=t_eval,
            t_original=t_original,
            y0=y0,
            temperature_arr=temperature_data,
            rainfall_arr=rainfall_data,
            param_dict=param_dict,
        )

        pred_normalized = solution.t()[0][1:]
        true_normalized = cumulative_cases[1:].squeeze()

        pred = pred_normalized.expm1()
        true = true_normalized.expm1()

        criterion = nn.MSELoss()
        loss_normalized = float(
            criterion(pred_normalized, true_normalized).item())
        loss_unnormalized = float(criterion(pred, true).item())

    # Figure 1: Normalized cases
    fig1, ax1 = plt.subplots()
    ax1.plot(true_normalized.cpu().numpy(), label='True Cases')
    ax1.plot(pred_normalized.cpu().numpy(), label='Predicted Cases')
    ax1.set_xlabel('Time (weeks)')
    ax1.set_ylabel('Normalized Cases')
    ax1.set_title('Projection: True vs Predicted Dengue Cases')
    ax1.text(0.5, -0.2, f'MSE Loss: {loss_normalized:.4f}',
             transform=ax1.transAxes, ha='center')
    ax1.legend()
    ax1.grid(True)
    fig1.savefig(f'{figure_dir}/projection_normalized_cases.png',
                 bbox_inches='tight')
    plt.close(fig1)

    # Figure 2: Cumulative cases
    fig2, ax2 = plt.subplots()
    ax2.plot(true.cpu().numpy(), label='True Cases')
    ax2.plot(pred.cpu().numpy(), label='Predicted Cases')
    ax2.set_xlabel('Time (weeks)')
    ax2.set_ylabel('Cumulative Cases')
    ax2.set_title('Projection: True vs Predicted Dengue Cases')
    ax2.text(0.5, -0.2, f'MSE Loss: {loss_unnormalized:.4f}',
             transform=ax2.transAxes, ha='center')
    ax2.legend()
    ax2.grid(True)
    fig2.savefig(f'{figure_dir}/projection_cumulative_cases.png',
                 bbox_inches='tight')
    plt.close(fig2)

    with open(f'{result_dir}/projection_loss_result.yaml', 'w') as f:
        yaml.safe_dump(
            {
                'projection_normalized_loss': loss_normalized,
                'projection_unnormalized_loss': loss_unnormalized,
            },
            f,
            sort_keys=False,
        )

    return {
        'projection_normalized_loss': loss_normalized,
        'projection_unnormalized_loss': loss_unnormalized,
    }


def train_wrapper(args):
    """Wrapper function for multiprocessing pool."""
    seed, location, device = args[:3]
    hyperparams = args[3] if len(args) > 3 else {}
    from src.utils import create_dir

    lr = hyperparams.get('lr', 1e-2)
    weight_decay = hyperparams.get('weight_decay', 0.0)
    beta1 = hyperparams.get('beta1', 0.9)
    beta2 = hyperparams.get('beta2', 0.999)

    dengue_nn = DengueNN(
        device=device,
        data_csv_path=f'data/calibration/{location}/data.csv',
        params_yaml_path='data/configs/params.yaml',
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
        epochs=500,
        hidden_dim=32,
        hidden_num=3,
    )

    result_dir = f'results/{location}/seed_{seed}'
    figure_dir = f'{result_dir}/figures'
    create_dir(result_dir)
    create_dir(figure_dir)

    run_name = f"{location}_seed_{seed}"
    with mlflow.start_run(run_name=run_name):
        # Log basic run params
        mlflow.log_param('seed', seed)
        mlflow.log_param('location', location)
        mlflow.log_param('device', device)
        mlflow.log_param('lr', dengue_nn.lr)
        mlflow.log_param('weight_decay', weight_decay)
        mlflow.log_param('beta1', beta1)
        mlflow.log_param('beta2', beta2)
        mlflow.log_param('epochs', dengue_nn.epochs)
        try:
            hidden_dim = dengue_nn.model.linears[0].out_features
        except Exception:
            hidden_dim = None
        mlflow.log_param('hidden_dim', hidden_dim)
        mlflow.log_param('hidden_num', len(
            [l for l in dengue_nn.model.linears if isinstance(l, nn.Linear)]) - 1)

        # Run training
        metrics = train(dengue_nn, seed, result_dir,
                        figure_dir, location=location)

        # Log early stopping indicator
        mlflow.set_tag('early_stopped', str(
            metrics.get('early_stopped', False)))
        mlflow.log_metric('early_stopped', int(
            metrics.get('early_stopped', False)))

        # Log calibration metrics
        mlflow.log_metric('calibration_best_loss', metrics.get('best_loss'))
        mlflow.log_metric('calibration_best_epoch',
                          metrics.get('best_epoch') or 0)
        mlflow.log_metric('calibration_final_loss',
                          metrics.get('final_loss') or 0)
        mlflow.log_metric('calibration_normalized_loss',
                          metrics.get('normalized_loss'))
        mlflow.log_metric('calibration_unnormalized_loss',
                          metrics.get('unnormalized_loss'))

        # Log artifacts: best params, model and figures
        best_params_file = os.path.join(
            metrics.get('result_dir'), 'best_params.yaml')
        if os.path.exists(best_params_file):
            mlflow.log_artifact(best_params_file, artifact_path='best_params')

        if os.path.exists(figure_dir):
            for fname in os.listdir(figure_dir):
                fpath = os.path.join(figure_dir, fname)
                if os.path.isfile(fpath) and not fname.startswith('projection_'):
                    mlflow.log_artifact(
                        fpath, artifact_path='calibration_figures')

        # Run projection with the calibrated params
        projection_metrics = run_projection(location, result_dir, figure_dir)
        if projection_metrics is not None:
            mlflow.log_metric('projection_normalized_loss',
                              projection_metrics['projection_normalized_loss'])
            mlflow.log_metric('projection_unnormalized_loss',
                              projection_metrics['projection_unnormalized_loss'])
            metrics.update(projection_metrics)
            proj_loss_file = os.path.join(
                result_dir, 'projection_loss_result.yaml')
            if os.path.exists(proj_loss_file):
                mlflow.log_artifact(proj_loss_file, artifact_path='projection')
            for fname in os.listdir(figure_dir):
                if fname.startswith('projection_'):
                    mlflow.log_artifact(os.path.join(
                        figure_dir, fname), artifact_path='projection_figures')

    return metrics
