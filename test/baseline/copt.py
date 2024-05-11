import coptpy as cp
from coptpy import COPT
import time

def set_model(model, show_log):
    model.setParam(COPT.Param.TimeLimit, 250)
    if show_log:
        model.setParam(COPT.Param.Logging, 1)
    else:
        model.setParam(COPT.Param.Logging, 0)

def solve_mps(mps_file, show_log=True):
    env = cp.Envr()
    model = env.createModel("test")
    model.readMps(mps_file)
    set_model(model, show_log)
    start_time = time.time()
    model.solve()
    return model, time.time() - start_time

if __name__ == '__main__':
    model, total_time = solve_mps("test/example/test_0.mps")
    print(f"Status: {model.status}")
    if model.status == COPT.OPTIMAL:
        print(f"Optimal value: {model.objval:}")
        print(f"Time: {model.getAttr('SolvingTime'):.2e} s")
        allVars = model.getVars()
        for var in allVars:
            print(f"{var.index+1} = {var.x}")