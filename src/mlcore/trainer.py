import os
import gc
import warnings
import random
import copy
import yaml

import torch
import torch.nn as nn
import numpy as np
from mlflow.data.pandas_dataset import PandasDataset
from tqdm import tqdm

from src.mlcore import DengueNN
from src.odecore import get_solution
from src.utils import *
from src.config import INITIAL_STATE_ORDER, MLFLOW_TRACKING_URI

_CALIBRATION_EXPERIMENT = 'dengue_calibration'


def train_wrapper(args):
    """Wrapper function for multiprocessing pool."""
    seed, location, device = args[:3]
    hyperparams = args[3] if len(args) > 3 else {}
    epochs = args[4]
    from src.utils import create_dir

    lr = hyperparams.get('lr', 1e-4)
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
        epochs=epochs,
        hidden_dim=32,
        hidden_num=3,
    )

    result_dir = f'results/{location}/seed_{seed}'
    figure_dir = f'{result_dir}/figures'
    create_dir(result_dir)
    create_dir(figure_dir)

    # Explicitly configure MLflow in each worker process.
    set_tracking_uri(MLFLOW_TRACKING_URI)
    set_experiment(_CALIBRATION_EXPERIMENT)

    run_name = f"{location}_seed_{seed}"
    with start_run(run_name=run_name):
        # Log basic run params
        log_params(
            {
                'seed': seed,
                'location': location,
                'device': device,
                'lr': dengue_nn.lr,
                'weight_decay': weight_decay,
                'beta1': beta1,
                'beta2': beta2,
                'epochs': dengue_nn.epochs,
            }
        )
        try:
            hidden_dim = dengue_nn.model.linears[0].out_features
        except Exception:
            hidden_dim = None
        log_params(
            {
                'hidden_dim': hidden_dim,
                'hidden_num': len(
                    [l for l in dengue_nn.model.linears if isinstance(
                        l, nn.Linear)]
                ) - 1,
            }
        )

        # Log dataset and params YAML used for this run as artifacts
        try:
            log_artifact_if_exists(
                dengue_nn.data_csv_path,
                artifact_path='dataset',
            )
        except Exception:
            pass
        try:
            log_artifact_if_exists(
                dengue_nn.params_yaml_path,
                artifact_path='params',
            )
        except Exception:
            pass

        # Run training
        metrics = train(dengue_nn, seed, result_dir,
                        figure_dir, location=location)

        # Log artifacts: model and figures
        log_dir_artifacts(
            figure_dir,
            artifact_path='calibration_figures',
            exclude_prefix='projection_',
        )

        # Run projection with the calibrated params
        best_params_for_projection = {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in (metrics.get('best_params') or {}).items()
        }
        projection_metrics = run_projection(
            location, result_dir, figure_dir, best_params_for_projection)
        if projection_metrics is not None:
            metrics.update(projection_metrics)
            proj_loss_file = os.path.join(
                result_dir, 'projection_loss_result.yaml')
            log_artifact_if_exists(proj_loss_file, artifact_path='projection')
            log_dir_artifacts(
                figure_dir,
                artifact_path='projection_figures',
                prefix='projection_',
            )

    # Clean up memory for this wrapper run
    del dengue_nn
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f'Memory released after training for {location} seed {seed}')


def train(dengue_model, seed, result_dir, figure_dir, location='unknown'):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        dengue_nn = copy.deepcopy(dengue_model)
        dengue_nn.model.train()
        data = dengue_nn.normalized_data[0]
        data_df = pd.read_csv(dengue_nn.data_csv_path)
        dataset: PandasDataset = mlflow.data.from_pandas(data_df)
        mlflow.log_input(dataset, context='training')

        loss_history = []
        best_loss = float('inf')
        best_epoch = None
        best_solution = None
        best_param_dict = {}
        best_model_state = None

        early_stopping_patience = 20
        early_stopping_min_delta = 1e-4
        early_stopping_warmup = 50
        epochs_without_improvement = 0
        early_stopped = False

        # Training loop
        progress_bar = tqdm(
            range(dengue_nn.epochs),
            desc='Training',
            leave=False,
        )
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

            y0_list = [data]
            for state in outputs[:len(INITIAL_STATE_ORDER)]:
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
            true_solution = dengue_nn.normalized_data[1:].squeeze()

            loss = dengue_nn.criterion(predicted_solution, true_solution)
            loss.backward()
            dengue_nn.optimizer.step()
            dengue_nn.scheduler.step()
            loss_history.append(loss.item())
            progress_bar.set_postfix(
                loss=f'{loss.item():.6f}',
                best_loss=f'{best_loss:.6f}' if best_loss != float(
                    'inf') else 'N/A',
            )

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
                        print(
                            f'Early stopping at epoch {epoch+1}/{dengue_nn.epochs}, Best Loss: {best_loss:.6f}')
                        break

        progress_bar.close()

        if early_stopped:
            print(
                f'Training finished early at epoch {epoch+1}. Best epoch: {best_epoch}, Best loss: {best_loss:.6f}')
        else:
            print(
                f'Training finished after {dengue_nn.epochs} epochs. Best epoch: {best_epoch}, Best loss: {best_loss:.6f}')

        # Create and save figures after training
        save_calibration_figures(
            dengue_nn.normalized_data,
            best_solution,
            best_loss,
            loss_history,
            figure_dir,
        )

        # Save calibrated params as YAML and log as an artifact.
        calibrated_params_path = os.path.join(
            result_dir, 'calibrated_params.yaml')
        with open(calibrated_params_path, 'w') as f:
            yaml.safe_dump(best_param_dict, f, sort_keys=False)
        log_artifact_if_exists(calibrated_params_path,
                               artifact_path='calibration')

        # Save results
        with open(f'{result_dir}/loss_result.yaml', 'w') as f:
            yaml.safe_dump(
                {
                    'best_epoch': int(best_epoch),
                    'loss': float(loss),
                },
                f,
                sort_keys=False,
            )

        best_epoch_value = int(best_epoch) if best_epoch is not None else 0
        best_loss_value = float(best_loss) if best_loss != float(
            'inf') else float(loss)

        set_tag('early_stopped', early_stopped)
        log_param('early_stopped', early_stopped)
        log_metrics(
            {
                'best_epoch': best_epoch_value,
                'best_loss': best_loss_value,
            }
        )

        dengue_nn.param_df.to_csv(
            f'{result_dir}/param_history.csv', index=False)
        if best_model_state is not None:
            dengue_nn.model.load_state_dict(best_model_state)
            log_pytorch_model(
                dengue_nn.model,
                name='dengue_nn',
                registered_model_name=f'dengue_nn_{location}_seed_{seed}',
            )

        # Clean up memory
        del dengue_nn
        del data
        del best_solution
        del loss_history
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        plt.close('all')

    return {
        'result_dir': result_dir,
        'best_epoch': int(best_epoch) if best_epoch is not None else None,
        'loss': float(loss),
        'early_stopped': early_stopped,
        'best_params': best_param_dict,
    }


def run_projection(location, result_dir, figure_dir, param_dict):
    """Run forward projection using calibrated params on the projection dataset."""
    data_csv_path = f'data/projection/{location}/data.csv'

    if not os.path.exists(data_csv_path) or not param_dict:
        return None

    with torch.no_grad():
        cumulative_cases = extract_cumulative_cases(data_csv_path)
        normalized_data = cumulative_cases.log1p()
        temperature_data, rainfall_data = extract_temperature_rainfall(
            data_csv_path)

        t_original = torch.linspace(0, 1, steps=len(
            cumulative_cases), dtype=torch.float32)
        t_eval = torch.linspace(0, 1, steps=len(
            cumulative_cases), dtype=torch.float32)

        y0_list = [normalized_data[0]]
        for state_name in INITIAL_STATE_ORDER:
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

        pred = solution.t()[0][1:]
        true = normalized_data[1:].squeeze()
        loss = nn.MSELoss()(pred, true).item()

        save_projection_figures(
            normalized_data.detach().cpu().numpy(),
            solution.t()[0].detach().cpu().numpy(),
            figure_dir,
        )

    with open(f'{result_dir}/projection_loss_result.yaml', 'w') as f:
        yaml.safe_dump(
            {
                'projection_loss': loss,
            },
            f,
            sort_keys=False,
        )

    projection_result = {
        'projection_loss': loss,
    }

    log_metrics(
        {
            'projection_loss': loss,
        }
    )

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f'Memory released after projection for {location}')

    return projection_result
