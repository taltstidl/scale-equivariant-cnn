"""
Script to clean exported runs.
"""
import numpy as np
import pandas as pd

all_datasets = ['emoji', 'mnist', 'trafficsign', 'aerial']
all_models = ['standard', 'pixel_pool', 'energy_pool', 'slice_pool', 'conv3d', 'ensemble', 'spatial_transform', 'xu',
              'kanazawa', 'hermite', 'disco']


def check_base_runs():
    """ Checks whether 5 runs each exist for learning rates 1e-2 and 1e-3. """
    runs = pd.read_csv('runs.csv')
    runs = runs[runs['status'] == 'FINISHED']  # Filter for finished runs
    runs = runs[runs['metrics.test_acc'].notnull()]  # Filter for finished runs
    for dataset in all_datasets:
        for model in all_models:
            # Check how many different seeds were trained
            sub_runs = runs[(runs['params.model'] == model) & (runs['params.data'] == dataset)]
            for evaluation in [1, 2, 3, 4]:
                runs_1e2 = sub_runs[(sub_runs['params.evaluation'] == evaluation) & (sub_runs['params.lr'] == 1e-2)]
                runs_1e3 = sub_runs[(sub_runs['params.evaluation'] == evaluation) & (sub_runs['params.lr'] == 1e-3)]
                num_runs_1e2, num_runs_1e3 = runs_1e2.shape[0], runs_1e3.shape[0]
                if num_runs_1e2 < 5:
                    s = num_runs_1e2 + 1
                    print('sbatch train.sh {} {} {} 7 bicubic 1e-2 {} 5'.format(model, dataset, evaluation, s))
                if num_runs_1e3 < 5:
                    s = num_runs_1e3 + 1
                    print('sbatch train.sh {} {} {} 7 bicubic 1e-3 {} 5'.format(model, dataset, evaluation, s))


def check_selected_runs():
    """ Checks whether 50 (emoji) or 25 (other) runs each exist for best learning rates. """
    runs = pd.read_csv('runs.csv')
    runs = runs[runs['status'] == 'FINISHED']  # Filter for finished runs
    runs = runs[runs['metrics.test_acc'].notnull()]  # Filter for finished runs
    runs_collected = []
    for dataset in all_datasets:
        for model in all_models:
            # Check how many different seeds were trained
            sub_runs = runs[(runs['params.model'] == model) & (runs['params.data'] == dataset)]
            for evaluation in [1, 2, 3, 4]:
                # Retrieve the first 5 runs (seeds 1 through 5) for 1e-2 and 1e-3
                runs_1e2 = sub_runs[(sub_runs['params.evaluation'] == evaluation) & (sub_runs['params.lr'] == 1e-2)]
                runs_1e2_first5 = runs_1e2.tail(5)  # Get first 5 runs
                runs_1e3 = sub_runs[(sub_runs['params.evaluation'] == evaluation) & (sub_runs['params.lr'] == 1e-3)]
                runs_1e3_first5 = runs_1e3.tail(5)  # Get first 5 runs
                # Compute median validation performance for 1e-2 and 1e-3
                perf_1e2 = np.median(runs_1e2_first5['metrics.best_acc'].to_numpy())
                perf_1e3 = np.median(runs_1e3_first5['metrics.best_acc'].to_numpy())
                # Find the best learning rate
                runs_selected = runs_1e2 if perf_1e2 > perf_1e3 else runs_1e3
                num_runs_selected = runs_selected.shape[0]
                # Check whether the targeted number of runs exists
                num_runs_targeted = 50 if dataset == 'emoji' else 25
                if num_runs_targeted > num_runs_selected:
                    s, t = num_runs_selected + 1, num_runs_targeted
                    lr = '1e-2' if perf_1e2 > perf_1e3 else '1e-3'
                    print('sbatch train.sh {} {} {} 7 bicubic {} {} {}'.format(model, dataset, evaluation, lr, s, t))
                runs_collected.append(runs_selected)
    clean = pd.concat(runs_collected)
    clean.to_csv('clean.csv', index=False)


def main():
    # check_base_runs()
    check_selected_runs()


if __name__ == '__main__':
    main()
