import yaml
import argparse
import multiprocessing as mp

import cProfile
import mlflow

from src.mlcore import (
    train_wrapper,
    tune_wrapper,
    load_tuned_params,
)
from src.config import MLFLOW_TRACKING_URI


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Dengue NN calibration / tuning runner.')
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Run hyperparameter tuning with Optuna instead of calibration.',
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of Optuna trials per location (tuning mode only).',
    )
    parser.add_argument(
        '--tune-epochs',
        type=int,
        default=50,
        help='Training epochs per Optuna trial (tuning mode only).',
    )
    parser.add_argument(
        '--location',
        type=str,
        default=None,
        help='Restrict tuning/calibration to a single location.',
    )
    parser.add_argument(
        '--calibration-epochs',
        type=int,
        default=500,
        help='Training epochs for calibration runs (calibration mode only).',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='Number of worker processes for tuning/calibration. Defaults to 1. Use 1 for serial.',
    )
    parser.add_argument(
        '--use-default-hyperparams',
        action='store_true',
        help='Disable loading tuned hyperparameters and use built-in defaults.',
    )
    args = parser.parse_args()

    pr = cProfile.Profile()
    pr.enable()

    with open('data/configs/weather_data_params.yaml', 'r') as file:
        location_params = yaml.safe_load(file)
    location_list = list(location_params['location'].keys())
    if args.location:
        location_list = [args.location]

    device = 'cpu'

    if args.tune:
        print(f'Running hyperparameter tuning for: {location_list}')
        tune_tasks = [
            (location, device, args.tune_epochs, args.n_trials)
            for location in location_list
        ]

        num_workers = max(1, args.num_workers)

        print(
            f'Running tuning with {num_workers} worker(s) across {len(tune_tasks)} location(s).', flush=True)

        if num_workers == 1:
            for task in tune_tasks:
                tune_wrapper(task)
                print(
                    f'Completed tuning for location: {task[0]}',
                    flush=True,
                )
        else:
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=num_workers) as pool:
                async_results = [pool.apply_async(
                    tune_wrapper, args=(task,)) for task in tune_tasks]
                pool.close()
                pool.join()
                for res in async_results:
                    res.get()
    else:
        random_seed = [1, 2, 42, 1234, 1337, 9173, 6164, 5956, 7443, 1354]
        # random_seed = [0]

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f'MLflow tracking URI: {MLFLOW_TRACKING_URI}')
        mlflow.set_experiment('dengue_calibration')

        # Load tuned hyperparameters for each location (falls back to defaults if absent)
        location_hyperparams = {}
        for location in location_list:
            if args.use_default_hyperparams:
                location_hyperparams[location] = {}
                print(
                    f'Using default hyperparameters for {location} (tuned hyperparameters disabled).'
                )
                continue

            params = load_tuned_params(
                location,
                mlflow_tracking_uri=MLFLOW_TRACKING_URI,
                mlflow_experiment='dengue_nn_tuning',
            )
            location_hyperparams[location] = params
            if params:
                print(f'Loaded tuned params for {location}: {params}')
            else:
                print(f'No tuned params found for {location}, using defaults.')

        tasks = [(seed, location, device, location_hyperparams[location], args.calibration_epochs)
                 for seed in random_seed for location in location_list]

        num_workers = max(1, args.num_workers)

        print(
            f'Running calibration with {num_workers} worker(s) across {len(tasks)} task(s).', flush=True)

        if num_workers == 1:
            for task in tasks:
                train_wrapper(task)
                print(
                    f'Completed calibration for location: {task[1]} with seed: {task[0]}',
                    flush=True,
                )
        else:
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=num_workers) as pool:
                async_results = [pool.apply_async(
                    train_wrapper, args=(task,)) for task in tasks]
                pool.close()
                pool.join()
                for res in async_results:
                    res.get()

    print('Finished', flush=True)

    pr.disable()
    pr.dump_stats('profile_results.prof')
