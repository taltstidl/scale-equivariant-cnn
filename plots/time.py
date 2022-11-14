import re

import numpy as np
import pandas as pd


def write_table(runs):
    runs = runs[runs['params.evaluation'] == 2]
    table = pd.pivot_table(runs, values='metrics.train_time', index='params.model', columns='params.data',
                           aggfunc=np.mean, sort=False)
    table = table[['emoji', 'mnist', 'trafficsign', 'aerial']]
    latex = table.to_latex(float_format="{:.2f}".format)
    print(re.sub(r"(?:(?!\n)\s)+", " ", latex))


def main():
    # Write table to console
    runs = pd.read_csv('../scripts/clean.csv')
    write_table(runs)


if __name__ == '__main__':
    main()
