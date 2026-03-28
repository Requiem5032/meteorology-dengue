import gc
import copy
import warnings
import random

import optuna
import torch
import numpy as np
from tqdm import tqdm

from src.mlcore import DengueNN
from src.odecore import get_solution
from src.config import MLFLOW_TRACKING_URI, INITIAL_STATE_ORDER
from src.utils import (
    log_artifact_if_exists,
    log_metric,
    log_metrics,
    log_param,
    log_params,
    search_runs,
    set_experiment,
    set_tracking_uri,
    start_run,
)


def tune_wrapper(args):
    location, device, tune_epochs, n_trials = args
    tune(
        location=location,
        device=device,
        epochs=tune_epochs,
        n_trials=n_trials,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        mlflow_experiment='dengue_nn_tuning',
    )


def tune(
    location,
    device='cpu',
    seed=42,
    epochs=None,
    n_trials=None,
    study_name=None,
    data_csv_path=None,
    params_yaml_path='data/configs/params.yaml',
    lr_range=(1e-6, 1e-2),
    weight_decay_range=(1e-6, 1e-2),
    beta1_range=(0.8, 0.99),
    beta2_range=(0.9, 0.9999),
    direction='minimize',
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    mlflow_experiment='dengue_nn_tuning',
    trial_patience=10,
    trial_min_delta=1e-6,
    study_patience=20,
):
    """Tune hyperparameters of the DengueNN model with Optuna.

    Parameters
    ----------
    location : str
        Location name (e.g. 'bello').  Used to derive ``data_csv_path`` when
        that argument is *None*.
    device : str
        PyTorch device string.
    seed : int
        Random seed used during each trial's training run.
    epochs : int
        Number of training epochs per trial.
    n_trials : int
        Number of Optuna trials.
    study_name : str or None
        Name of the Optuna study.  Defaults to ``f'dengue_nn_{location}'``.
    data_csv_path : str or None
        Path to the calibration CSV.  Defaults to
        ``f'data/calibration/{location}/data.csv'``.
    params_yaml_path : str
        Path to the parameters YAML file.
    lr_range : tuple[float, float]
        (low, high) for the learning-rate log-uniform search.
    weight_decay_range : tuple[float, float]
        (low, high) for the weight-decay log-uniform search.
    beta1_range : tuple[float, float]
        (low, high) for Adam beta1 (uniform search).
    beta2_range : tuple[float, float]
        (low, high) for Adam beta2 (uniform search).
    direction : str
        Optimisation direction passed to ``optuna.create_study``.
    mlflow_tracking_uri : str
        MLflow tracking URI.  Defaults to a local SQLite DB.
    mlflow_experiment : str
        MLflow experiment name for tuning runs.
    trial_patience : int
        Epochs without improvement before a trial is stopped early (default 10).
    trial_min_delta : float
        Minimum loss decrease to count as an improvement within a trial.
    study_patience : int
        Number of consecutive trials without improvement before the whole
        study is stopped early (default 15).

    Returns
    -------
    optuna.study.Study
        The completed Optuna study object.  Access the best params via
        ``study.best_params`` and the best value via ``study.best_value``.
    """
    if data_csv_path is None:
        data_csv_path = f'data/calibration/{location}/data.csv'
    if study_name is None:
        study_name = f'dengue_nn_{location}'

    set_tracking_uri(mlflow_tracking_uri)
    set_experiment(mlflow_experiment)

    def objective(trial):
        lr = trial.suggest_float('lr', lr_range[0], lr_range[1], log=True)
        weight_decay = trial.suggest_float(
            'weight_decay', weight_decay_range[0], weight_decay_range[1], log=True)
        beta1 = trial.suggest_float('beta1', beta1_range[0], beta1_range[1])
        beta2 = trial.suggest_float('beta2', beta2_range[0], beta2_range[1])

        # Build model
        dengue_nn = DengueNN(
            device=device,
            data_csv_path=data_csv_path,
            params_yaml_path=params_yaml_path,
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            epochs=epochs,
            hidden_dim=32,
            hidden_num=3,
        )

        best_loss = _run_trial_training(
            dengue_nn, epochs, seed,
            trial=trial,
            patience=trial_patience,
            min_delta=trial_min_delta,
        )

        del dengue_nn
        gc.collect()

        # Log this trial as a nested MLflow child run
        with start_run(run_name=f'{study_name}_trial_{trial.number}', nested=True):
            # Log dataset and params YAML used for this trial
            try:
                log_artifact_if_exists(data_csv_path, artifact_path='dataset')
            except Exception:
                pass
            try:
                log_artifact_if_exists(
                    params_yaml_path,
                    artifact_path='params',
                )
            except Exception:
                pass

            log_params(
                {
                    'location': location,
                    'seed': seed,
                    'epochs': epochs,
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'beta1': beta1,
                    'beta2': beta2,
                }
            )
            log_metric('best_loss', best_loss)

        return best_loss

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=20, n_warmup_steps=10)
    study = optuna.create_study(
        study_name=study_name, direction=direction, sampler=sampler, pruner=pruner)

    stopper = _StudyEarlyStopper(patience=study_patience)

    with start_run(run_name=study_name):
        log_params(
            {
                'location': location,
                'seed': seed,
                'epochs': epochs,
                'n_trials': n_trials,
                'direction': direction,
                'trial_patience': trial_patience,
                'study_patience': study_patience,
            }
        )

        # Log dataset and params YAML used for the tuning study
        try:
            log_artifact_if_exists(data_csv_path, artifact_path='dataset')
        except Exception:
            pass
        try:
            log_artifact_if_exists(params_yaml_path, artifact_path='params')
        except Exception:
            pass

        study.optimize(objective, n_trials=n_trials, callbacks=[
                       stopper], show_progress_bar=True)

        # Log best trial summary to the parent run
        log_params({f'best_{k}': v for k, v in study.best_params.items()})
        log_metrics({'best_loss': study.best_value})
        log_param('best_trial_number', study.best_trial.number)

    print(f'\nBest trial for {location}:')
    print(f'Value (best loss): {study.best_value:.6f}')
    print('Params:')
    for k, v in study.best_params.items():
        print(f'{k}: {v}')

    return study


def load_tuned_params(
    location,
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    mlflow_experiment='dengue_nn_tuning',
):
    """Load the best hyperparameters for *location* from the MLflow tuning runs.

    Returns a dict with keys ``lr``, ``weight_decay``, ``beta1``, ``beta2``,
    or an empty dict if no completed tuning run is found.
    """
    set_tracking_uri(mlflow_tracking_uri)
    study_name = f'dengue_nn_{location}'

    try:
        runs = search_runs(
            experiment_names=[mlflow_experiment],
            filter_string=f"tags.`mlflow.runName` = '{study_name}'",
            order_by=['start_time DESC'],
            max_results=10,
        )
    except Exception:
        return {}

    # Keep only parent runs (child runs have mlflow.parentRunId tag)
    parent_runs = runs[~runs['tags.mlflow.parentRunId'].notna(
    )] if 'tags.mlflow.parentRunId' in runs.columns else runs
    if parent_runs.empty:
        return {}

    row = parent_runs.iloc[0]
    keys = ('lr', 'weight_decay', 'beta1', 'beta2')
    params = {}
    for k in keys:
        col = f'params.best_{k}'
        if col in row and row[col] is not None:
            try:
                params[k] = float(row[col])
            except (TypeError, ValueError):
                pass

    return params


def _run_trial_training(dengue_nn, epochs, seed, trial=None, patience=20, min_delta=1e-6):
    """Run a single training loop and return the best loss achieved."""

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = copy.deepcopy(dengue_nn)
        model.model.train()
        data = model.normalized_data[0]

        best_loss = float('inf')
        no_improve_count = 0

        progress_bar = tqdm(range(epochs), desc='Trial', leave=False)
        for epoch in progress_bar:
            model.optimizer.zero_grad()
            outputs = model.model(data)

            t_original = torch.linspace(
                0, 1,
                steps=len(model.normalized_data),
                dtype=torch.float32,
                device=model.device,
            )
            t_eval = torch.linspace(
                0, 1,
                steps=len(model.normalized_data),
                dtype=torch.float32,
                device=model.device,
            )

            for key, val in zip(model.learnable_params, outputs):
                model.param_dict[key] = val

            y0_list = [data]
            for state in outputs[:len(INITIAL_STATE_ORDER)]:
                y0_list.append(torch.atleast_1d(state))
            y0 = torch.stack(y0_list).squeeze(-1)

            solution = get_solution(
                t_eval=t_eval,
                t_original=t_original,
                y0=y0,
                temperature_arr=model.temperature_data,
                rainfall_arr=model.rainfall_data,
                param_dict=model.param_dict,
            )

            predicted_solution = solution.t()[0][1:]
            true_solution = model.normalized_data[1:].squeeze()

            loss = model.criterion(predicted_solution, true_solution)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f'Non-finite loss at epoch {epoch+1}: {loss.item()}'
                )

            loss.backward()
            model.optimizer.step()
            model.scheduler.step()

            loss_val = loss.item()

            if loss_val < best_loss - min_delta:
                best_loss = loss_val
                no_improve_count = 0
            else:
                no_improve_count += 1

            progress_bar.set_description(
                f'Epoch {epoch+1}/{epochs}, Loss: {loss_val:.6f}')

            # Report intermediate value so Optuna pruner can evaluate it
            if trial is not None:
                trial.report(loss_val, epoch)
                if trial.should_prune():
                    del model
                    gc.collect()
                    raise optuna.exceptions.TrialPruned()

            # Within-trial early stopping
            if no_improve_count >= patience:
                progress_bar.set_description(
                    f'Early stop at epoch {epoch+1}, best loss: {best_loss:.6f}')
                break

        del model
        gc.collect()

    return best_loss


class _StudyEarlyStopper:
    """Optuna callback that stops the study when no improvement is seen
    for ``patience`` consecutive trials."""

    def __init__(self, patience: int):
        self.patience = patience
        self._no_improve = 0
        self._best = float('inf')

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if study.best_value < self._best:
            self._best = study.best_value
            self._no_improve = 0
        else:
            self._no_improve += 1
        if self._no_improve >= self.patience:
            print(
                f'\nStudy early stopping: no improvement for {self.patience} trials.')
            study.stop()
