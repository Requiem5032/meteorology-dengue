import os
import yaml
from collections import defaultdict
import pandas as pd  # Add pandas import

RESULTS_DIR = 'results'
COUNTRIES = ['bello', 'cambodia', 'iquitos', 'philippines', 'sanjuan', 'vietnam']

def collect_param_ranges():
    param_ranges = {}

    for country in COUNTRIES:
        country_dir = os.path.join(RESULTS_DIR, country)
        if not os.path.isdir(country_dir):
            continue

        # Find all seed directories (skip 'average')
        seed_dirs = [
            d for d in os.listdir(country_dir)
            if d.startswith('seed_') and os.path.isdir(os.path.join(country_dir, d))
        ]

        param_values = defaultdict(list)

        for seed in seed_dirs:
            param_path = os.path.join(country_dir, seed, 'best_params.yaml')
            if not os.path.isfile(param_path):
                continue
            with open(param_path, 'r') as f:
                params = yaml.safe_load(f)
                for k, v in params.items():
                    param_values[k].append(v)

        # Aggregate min/max for each param
        param_ranges[country] = {}
        for k, values in param_values.items():
            param_ranges[country][k] = {
                'min': min(values),
                'max': max(values)
            }

    return param_ranges

if __name__ == '__main__':
    ranges = collect_param_ranges()
    # Convert to DataFrame for saving
    rows = []
    for country, params in ranges.items():
        for param, stats in params.items():
            rows.append({
                'country': country,
                'param': param,
                'min': stats['min'],
                'max': stats['max']
            })
    df = pd.DataFrame(rows)
    df.to_csv('results/param_ranges.csv', index=False)
    print('Saved parameter ranges to results/param_ranges.csv')