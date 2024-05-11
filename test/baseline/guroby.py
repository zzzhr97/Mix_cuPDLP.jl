import gurobipy as grb
import time

def set_model(model, show_log):
    # 显示求解过程
    model.Params.LogToConsole=show_log     
    # 百分比界差
    # model.Params.MIPGap=0.0001             
    # 限制求解时间为 1000s
    model.Params.TimeLimit=1200
    # 预处理程度, 0关闭,1保守,2激进
    # model.Params.Presolve = -1 
    # 求解侧重点. 1快速找到可行解, 2证明最优, 3侧重边界提升, 0均衡搜索
    # model.Params.MIPFocus = 0 
    # 求解数量, 默认求所有解, 比较出最优的结果, 只需要可行解时可以设置该参数为1
    # model.Params.SolutionLimit = 1 
    # 默认求解器，改为 2 时可以解决非凸二次优化问题
    # model.Params.NonConvex = 1 

def solve_mps(mps_file, show_log=False):
    model = grb.read(mps_file)
    set_model(model, show_log)
    start_time = time.time()
    model.optimize()
    return model, time.time() - start_time


if __name__ == '__main__':
    model, total_time = solve_mps("test/example/test.mps")
    print(f"Optimal value: {model.objVal}")
    print(f"Status: {model.status}")
    print(f"Time: {total_time:.2e} s\n")