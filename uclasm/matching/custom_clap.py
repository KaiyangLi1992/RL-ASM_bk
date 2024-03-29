import numpy as np
from laptools import lap
from laptools import clap

def one_hot(idx, length):
    one_hot = np.zeros(length, dtype=np.bool)
    one_hot[idx] = True
    return one_hot

def naive_costs(cost_matrix):
    """An naive algorithm of solving all constraint LAPs. """
    n_rows, n_cols = cost_matrix.shape
    total_costs = np.full(cost_matrix.shape, np.inf)
    for i in range(n_rows):
        for j in range(n_cols):
            if cost_matrix[i,j] == np.inf:
                continue
            sub_row_ind = ~one_hot(i, n_rows)
            sub_col_ind = ~one_hot(j, n_cols)
            sub_cost_matrix = cost_matrix[sub_row_ind, :][:, sub_col_ind]
            row_idx, col_idx = lap.solve(sub_cost_matrix)
            sub_total_cost = sub_cost_matrix[row_idx, col_idx].sum()
            total_costs[i, j] = cost_matrix[i, j] + sub_total_cost
    return total_costs

cost = [[0. , np.inf, np.inf],
        [np.inf, 0. , 0.5],
        [np.inf, 1. , 0. ]]


cost = np.array(cost)
cost_clap1 = clap.costs(cost)
cost_clap2 = naive_costs(cost)

