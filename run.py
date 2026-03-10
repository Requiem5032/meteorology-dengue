import argparse
import cProfile
import multiprocessing as mp
import mlflow

from src.nn import *
from src.utils import *
from src.config import MLFLOW_TRACKING_URI


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dengue NN calibration / tuning runner.')
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
        for location in location_list:
            tune(
                location=location,
                device=device,
                epochs=args.tune_epochs,
                n_trials=args.n_trials,
                mlflow_tracking_uri=MLFLOW_TRACKING_URI,
                mlflow_experiment='dengue_nn_tuning',
            )
    else:
        random_seed = [0, 1, 42, 1234, 1337, 9173, 6164, 5956, 7443, 1354]
        # random_seed = [42]
        num_processes = 6
        print(f'Running calibration using {num_processes} processes.')

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f'MLflow tracking URI: {MLFLOW_TRACKING_URI}')
        mlflow.set_experiment('dengue_calibration')

        # Load tuned hyperparameters for each location (falls back to defaults if absent)
        location_hyperparams = {}
        for location in location_list:
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

        tasks = [(seed, location, device, location_hyperparams[location])
                 for seed in random_seed for location in location_list]

        pool = mp.Pool(processes=num_processes)
        try:
            results = pool.map(train_wrapper, tasks)
        finally:
            pool.close()
            pool.join()

    print('Finished', flush=True)

    pr.disable()
    pr.dump_stats('profile_results.prof')
