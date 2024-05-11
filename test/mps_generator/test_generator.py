from generator import *
from scipy.optimize import linprog
import numpy as np
import subprocess
import re
import time
import sys
import json

sys.path.append("test/baseline")
import test_baseline

# -19965.9737
# -24233.1075
# -5640.4345

shape = (15, 24)
only_generate = 0
test_cuPDLP = 0
test_mip_cuPDLP = 1
print_flag = 4
test_int = 2        # 0: no integer, 1: rand integer, 2: given
int_num = 8        # number of integer variables
prob_zero = 0.10    # probability of zero constraints Aij
time_sec_limit = 300

problem_num = 3
solve_num = 1

solver = 2          # 1: scipy, 2: copt
scales = {
    "x": (1, 16),
    "c": (-4, 4),
    "A": (-1, 1),
}

# 设置整数约束
def set_int(data):
    if test_int == 0:
        data["int_list"] = np.zeros(shape[1])
    elif test_int == 1:
        data["int_list"] = np.random.choice([0, 1], shape[1])
    elif test_int == 2:
        data["int_list"] = np.zeros(shape[1])
        data["int_list"][:int_num] = 1

# 设置零变量
def set_zero(data):
    mask = np.random.choice([0, 1], size=shape, p=[prob_zero, 1 - prob_zero])
    data["A"][mask == 0] = 0

def get_matrix(
        shape=(3, 5),
        scale=(0, 1),
        mat=None,
        dtype=float,
        rand=True,
):
    """Get numpy matrix."""
    a, b = scale
    if rand is True:
        return np.random.rand(*shape) * (b - a) + a
    else:
        return np.array(mat, dtype=dtype).reshape(shape)
    
def get_data(shape, scales, is_show):
    x = get_matrix(
        shape=(1, shape[1]),
        scale=scales['x'],
        rand=True,
    )
    data = {
        "c": get_matrix(
            shape=(shape[1],),
            scale=scales['c'],
            rand=True,
        ),
        "A": get_matrix(
            shape=shape,
            scale=scales['A'],
            rand=True,
        ),
        "b": None,
        "lb": np.zeros(shape[1]),
        "ub": get_matrix(
            shape=(shape[1],),
            scale=scales['x'],
            rand=True,
        ),
        "n_eq": np.random.randint(0, shape[0])
    }
    data['b'] = (data['A'] @ x.T).reshape(shape[0])

    set_int(data)
    set_zero(data)

    #rand_ub = np.ones(shape[1])
    rand_ub = np.random.randint(0, 2, shape[1])
    for i in range(shape[1]):
        if rand_ub[i] == 1:
            data['ub'][i] = np.nan
        elif data['int_list'][i] == 1:
            data['ub'][i] = np.ceil(data['ub'][i])

    if is_show:
        print("DATA:")
        print("<c>\n", data['c'])
        print("<A>\n", data['A'])
        print("<b>\n", data['b'])
        print("END")
    return data
    
def test_generator(solver, shape, scales, mps_file):
    if solver == 1:
        data = get_data(shape, scales, False)

        print("\n")
        rand_n = 0
        while 1:
            A_eq = data['A'][:data['n_eq']]
            b_eq = data['b'][:data['n_eq']]

            # Ax >= b --> -Ax <= -b
            A_ub = - data['A'][data['n_eq']:]
            b_ub = - data['b'][data['n_eq']:]
            c = data['c']
            bounds = [bound for bound in zip(data['lb'], data['ub'])]
            integrality = [i for i in data['int_list']]

            start_time = time.time()
            print(f'Start to solve {rand_n:04d}...', end='\r')
            options = {"disp": True, "time_limit": 1200}
            result = linprog(c, A_ub, b_ub, A_eq, b_eq, 
                            bounds=bounds, options=options, integrality=integrality)
            total_time = time.time() - start_time

            rand_n += 1
            print(f'Random generate {rand_n:04d}:', end=' ')
            pattern = r'HiGHS Status (\d+)'
            highs_status = int(re.findall(pattern, result.message)[0])
            print(result.message, end='\r')

            if result.success is True:
                print("\n")
                break

            data = get_data(shape, scales, False)

        write_matrix_to_mps(mps_file, data)
        print(result, "\n")
        print(f"Time: {total_time:2.2e} s\n")

    elif solver == 2:
        data = get_data(shape, scales, False)

        import copt

        print("\n")
        rand_n = 0
        while 1:
            print(f'Random generate {rand_n:04d}:')
            write_matrix_to_mps(mps_file, data)
            model, total_time = copt.solve_mps(mps_file)
            if model.status == copt.COPT.OPTIMAL and model.getAttr("NodeCnt") < 100000:
                print("End\n")
                break
            rand_n += 1
            data = get_data(shape, scales, False)

        result = model.objVal

    return total_time, result

def test_matrix():
    # 设置随机数种子
    timestamp = int(time.time())
    np.random.seed(timestamp)

    x = get_matrix(
        shape=(3, 5),
        scale=(0, 1),
        rand=True,
    )
    print(x)

def print_0(shape, total_time, result):
    iters = ["ZERO", "RAND", "FILE", "DIST 1e-5", 
            "FZERO 0.8", "FZERO 0.6", "FZERO 0.4", "FZERO 0.2",
            "FRAND 0.8", "FRAND 0.6", "FRAND 0.4", "FRAND 0.2"]
    for i in range(12):
        print(f"| {shape[0]} | {shape[1]} | {total_time:2.2e} |  |", end='')
        if i == 2:
            print(f" {result.fun:.4f} | 0 | {iters[i]} | 0 |")
        else:
            print(f" {result.fun:.4f} |  | {iters[i]} |  |")
    print("<!--  -->")
    print("  - `Test ({}, {})`".format(shape[0], shape[1]))
    for i in range(12):
        print("    - []")

def print_1(shape, total_time, result):
    print(f"| {shape[0],shape[1]} | {total_time:2.2e} |  |", end='')
    print(f" {result.fun:.4f} |  |  | `BEST` |")
    print(f"| {shape[0],shape[1]} | {total_time:2.2e} |  |", end='')
    print(f" {result.fun:.4f} |  |  | `BFS` |")
    print(f"| {shape[0],shape[1]} | {total_time:2.2e} |  |", end='')
    print(f" {result.fun:.4f} |  |  | `DFS` |")

def print_2(shape, total_time, result, baseline):
    if solver == 1:
        print(f"\nscipy: {result.fun}")
        print(f"scip: {baseline['scip'][0]}")
        print(f"guroby: {baseline['guroby'][0]}")
        print(f"copt: {baseline['copt'][0]}\n")

        print(f"| {shape[0]},{shape[1]} | {result.fun:.4f} | {total_time:2.2e} |", end="")
        print(f" {baseline['scip'][1]:.2e} | {baseline['guroby'][1]:.2e} ", end="")
        print(f"| {baseline['copt'][1]:.2e} |  |  |")
    elif solver == 2:
        print(f"\nscipy: \ ")
        print(f"scip: {baseline['scip'][0]}")
        print(f"guroby: {baseline['guroby'][0]}")
        print(f"copt: {baseline['copt'][0]}\n")

        print(f"| {shape[0]},{shape[1]} | {result:.4f} | \ |", end="")
        print(f" {baseline['scip'][1]:.2e} | {baseline['guroby'][1]:.2e} ", end="")
        print(f"| {baseline['copt'][1]:.2e} |  |  |")

def print_3(shape, baseline):
    print(f"\n| {shape[0]},{shape[1]} |  |  | {baseline['_scipy'][1]:.2e} ", end="")
    print(f"| {baseline['_scipy'][0]:.4f} | `_scipy` |")
    print(f"| {shape[0]},{shape[1]} |  |  | {baseline['scip'][1]:.2e} ", end="")
    print(f"| {baseline['scip'][0]:.4f} | `scip` |")
    print(f"| {shape[0]},{shape[1]} |  |  | {baseline['guroby'][1]:.2e} ", end="")
    print(f"| {baseline['guroby'][0]:.4f} | `guroby` |")
    print(f"| {shape[0]},{shape[1]} |  |  | {baseline['copt'][1]:.2e} ", end="")
    print(f"| {baseline['copt'][0]:.4f} | `copt` |")

def print_4(shape, baseline, copt_results):
    print(f"\n {shape[0]},{shape[1]}:")
    for solver in test_baseline.solvers:
        print(f"  - {solver}: {baseline[solver]/problem_num:.2e}")
    for result in copt_results:
        print(f"# {result:.4f}")

def test():
    baseline = {}
    base_mps_file = "test/example/test"
    copt_results = []
    for i in range(problem_num):
        cur_mps_file = base_mps_file + f"_{i}.mps"
        total_time, result = test_generator(
            solver, 
            shape, 
            scales,
            cur_mps_file
        )
        copt_results.append(result)
        print()
        print(">"*50)
        print(f"Problem {i}")
        print("<"*50)
        print()

        if only_generate == 1:
            exit()
        
        if print_flag in [2, 3]:
            baseline = test_baseline.test_baseline(cur_mps_file)
        elif print_flag in [4]:
            tmp = test_baseline.solve_mip_multi(cur_mps_file, solve_num)
            for key in tmp.keys():
                if key not in baseline.keys():
                    baseline[key] = 0
                baseline[key] += tmp[key]

    if print_flag == 0:
        print_0(shape, total_time, result)
    elif print_flag == 1:
        print_1(shape, total_time, result)
    elif print_flag == 2:
        print_2(shape, total_time, result, baseline)
    elif print_flag == 3:
        print_3(shape, baseline)
    elif print_flag == 4:
        print_4(shape, baseline, copt_results)
    

    print()
    input("Enter to continue...")

    if test_cuPDLP == 1 or test_mip_cuPDLP == 1:

        if test_cuPDLP == 1:
            command = [
                "julia",
                "--project",
                "./scripts/solve.jl",
                "--instance_path=test/example/test.mps",
                "--output_directory=tmp/test_solve",
                "--tolerance=1e-7",
                f"--time_sec_limit={time_sec_limit}"
            ]

        if test_mip_cuPDLP == 1:
            command = [
                "julia",
                "--project",
                "./scripts/solve_mip.jl",
                "--instance_path=test/example/test.mps",
                "--output_directory=tmp/test_solve",
                "--tolerance=1e-7",
                f"--time_sec_limit={time_sec_limit}"
            ]

        print("Begin to run command...\n")

        start_time = time.time()
        try:
            subprocess.run(command, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Running command {' '.join(command)} error: {e.returncode}")
        run_time = time.time() - start_time
        print(f"\nTime: {run_time:2.2e} s\n")

if __name__ == '__main__':
    test()