from pyscipopt import Model
import time

def solve_mps(mps_file, show_log=True):
    model = Model("test")
    model.readProblem(mps_file)
    if show_log is False:
        model.hideOutput(quiet=True)

    start_time = time.time()
    model.optimize()
    return model, time.time() - start_time
    model.getBestSol()
    model.getObjVal()
    model.getSolvingTime()
    model.getStatus()

if __name__ == '__main__':
    model, total_time = solve_mps("test/example/test.mps")
    print(f"Status: {model.getStatus()}")
    print(f"Optimal value: {model.getObjVal()}")
    print(f"Time: {model.getSolvingTime():.2e} s | {total_time:.2e} s")