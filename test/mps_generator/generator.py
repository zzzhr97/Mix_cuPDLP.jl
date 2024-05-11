import numpy as np

def write_comment(f, data):
    n = len(data['c'])

    f.write("* Integer variables:\n")
    for i in range(n):
        if i % 5 == 0:
            f.write("* ")

        f.write(f"x{i+1:04}:{data['int_list'][i]} ")

        if i % 5 == 4:
            f.write("\n")
    f.write("\n")

def write_obj_and_restriction(f, data):
    m = len(data['A'])
    n = len(data['c'])
    n_eq = data['n_eq']

    f.write("ROWS\n")
    f.write(" N  obj\n")

    for i in range(n_eq):
        f.write(f" E  E{i+1:04}\n")
    for i in range(m - n_eq):
        f.write(f" G  G{i+1:04}\n")
    
    f.write("COLUMNS\n")
    int_idx = 1
    for j in range(n):
        if data['int_list'][j] == 1:
            f.write(f"    MARKER{int_idx:04} \t 'MARKER' \t 'INTORG'\n")

        f.write(f"     x{j+1:04} \t obj  \t {data['c'][j]}\n")
        for i in range(n_eq):
            f.write(f"     x{j+1:04} \t E{i+1:04} \t {data['A'][i][j]}\n")
        for i in range(m - n_eq):
            f.write(f"     x{j+1:04} \t G{i+1:04} \t {data['A'][i+n_eq][j]}\n")

        if data['int_list'][j] == 1:
            f.write(f"    MARKER{int_idx:04} \t 'MARKER' \t 'INTEND'\n")
            int_idx += 1

    f.write("RHS\n")
    for i in range(n_eq):
        f.write(f"    rhs \t E{i+1:04} \t {data['b'][i]}\n")
    for i in range(m - n_eq):
        f.write(f"    rhs \t G{i+1:04} \t {data['b'][i+n_eq]}\n")

def write_bounds(f, data):
    n = len(data['c'])
    f.write("BOUNDS\n")
    for j in range(n):
        f.write(f" LO bounds x{j+1:04} \t {data['lb'][j]}\n")
        if not np.isnan(data['ub'][j]):
            f.write(f" UP bounds x{j+1:04} \t {data['ub'][j]}\n")

def write_matrix_to_mps(filename, data, is_show=True):
    """
    ## 将矩阵数据写入到.mps文件中
    - 格式要求
        - min cx
        - 前 n_eq 个式子:
            Ax = b
        - 后 m-n_eq 个式子:
            Ax >= b
        - lb <= x <= ub
            - ub 为 None 时，表示 x 无上界
        - int_list 为整数变量的索引列表，为 1 表示整数，为 0 表示非整数
    - 数据格式示例
        - data: {
            "c": [1, 2, 3],
            "A": [[1, 2, 3], [4, 5, 6]],
            "b": [1, 2],
            "lb": [0, 1.2, 3],
            "ub": [1.4, 2, None],
            "n_eq": 2,
            "int_list": [0, 1, 0],
        }
    """
    try:
        with open(filename, 'w') as f:
            write_comment(f, data)
            f.write(f"NAME {filename}\n")
            write_obj_and_restriction(f, data)
            write_bounds(f, data)
            f.write("ENDATA\n")
        
        if is_show:
            print(f"Writing .mps file {filename} successfully.")

    except Exception as e:
        print("Writing .mps error: ", str(e))

if __name__ == "__main__":
    data = {
        "c": [2.2, -1.4, 4.2, -6],
        "A": [[1, 2.2, 3, 5], [0.6, 3, 2.4, 7]],
        "b": [1, 2],
        "lb": [0, 0.1, 1, 2],
        "ub": [1.4, 2, 3, None],
        "n_eq": 2, 
        "int_list": [0, 1, 0, 1],
    }
    write_matrix_to_mps("example/test.mps", data)
