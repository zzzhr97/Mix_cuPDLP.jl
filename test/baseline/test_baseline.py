import copt
import guroby
import scip
import _scipy

show_log = False
#solvers = ['_scipy', 'scip', 'guroby', 'copt']
solvers = ['scip', 'guroby', 'copt']

def solve_mip(mps_file):
    result = {}
    for solver in solvers:
        print("="*40 + solver + "="*40)
        result[solver] = eval(solver + f".solve_mps('{mps_file}', {show_log})")
    return result

def solve_mip_multi(mps_file, num):
    result = {}
    for solver in solvers:
        cur_time = 0.0
        for i in range(num):
            cur_time += eval(solver + f".solve_mps('{mps_file}', {show_log})")[1]
        result[solver] = (cur_time / num)
    return result

def test_baseline(mps_file):
    result = solve_mip(mps_file)
    baseline = {}
    baseline['copt'] = (result['copt'][0].objval, result['copt'][1])
    baseline['guroby'] = (result['guroby'][0].objVal, result['guroby'][1])
    baseline['scip'] = (result['scip'][0].getObjVal(), result['scip'][1])
    baseline['_scipy'] = (result['_scipy'][0].fun, result['_scipy'][1])
    return baseline

if __name__ == '__main__':
    baseline = test_baseline("test/example/test.mps")
    print(baseline)

