import os
from contextlib import contextmanager

import mlflow


def set_tracking_uri(uri):
	mlflow.set_tracking_uri(uri)


def set_experiment(name):
	mlflow.set_experiment(name)


@contextmanager
def start_run(run_name=None, nested=False):
	with mlflow.start_run(run_name=run_name, nested=nested):
		yield


def set_tag(key, value):
	mlflow.set_tag(key, value)


def log_params(params):
	if not params:
		return
	mlflow.log_params(params)


def log_param(key, value):
	mlflow.log_param(key, value)


def log_metrics(metrics):
	if not metrics:
		return
	mlflow.log_metrics(metrics)


def log_metric(key, value):
	mlflow.log_metric(key, value)


def log_artifact_if_exists(path, artifact_path=None):
	if path and os.path.exists(path):
		mlflow.log_artifact(path, artifact_path=artifact_path)


def log_dir_artifacts(directory, artifact_path=None, prefix=None, exclude_prefix=None):
	if not directory or not os.path.exists(directory):
		return

	for fname in os.listdir(directory):
		if prefix is not None and not fname.startswith(prefix):
			continue
		if exclude_prefix is not None and fname.startswith(exclude_prefix):
			continue

		fpath = os.path.join(directory, fname)
		if os.path.isfile(fpath):
			mlflow.log_artifact(fpath, artifact_path=artifact_path)


def log_pytorch_model(model, name, registered_model_name=None):
	mlflow.pytorch.log_model(
		model,
		name=name,
		registered_model_name=registered_model_name,
	)


def search_runs(**kwargs):
	return mlflow.search_runs(**kwargs)
