


from pysmps import smps_loader as smps
from scipy.optimize import linprog
import numpy as np
import time

def read_mps(mps_file):
    mps = smps.load_mps(mps_file)
    n_eq = np.sum(np.array(mps[5]) == 'E')
    A_eq = np.array(mps[7][:n_eq])
    b_eq = np.array(mps[9]['rhs'][:n_eq])
    A_up = - np.array(mps[7][n_eq:])
    b_up = - np.array(mps[9]['rhs'][n_eq:])
    c = np.array(mps[6])
    bounds = [bound for bound in zip(mps[11]['bounds']['LO'], mps[11]['bounds']['UP'])]
    integrality = np.zeros(len(c))
    
    integrality[np.array(mps[4]) == 'integral'] = 1
    return A_eq, b_eq, A_up, b_up, c, bounds, integrality

def solve_mps(mps_file, show_log=False):
    A_eq, b_eq, A_up, b_up, c, bounds, integrality = read_mps(mps_file)
    options = {"disp": show_log, "time_limit": 2400}
    start_time = time.time()
    print("Begin to solve by scipy...")
    result = linprog(c, A_up, b_up, A_eq, b_eq, options=options,
                    bounds=bounds, integrality=integrality, method='highs')
    total_time = time.time() - start_time
    return result, total_time

if __name__ == '__main__':
    result, total_time = solve_mps("test/example/test.mps")
    print(result, "\n")
    print(f"Time: {total_time:2.2e} s\n")
    print(result.fun)