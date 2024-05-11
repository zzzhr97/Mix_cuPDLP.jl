
"""
不同初始化的方式
- INIT_TYPE_ZERO: 全零
- INIT_TYPE_ONES: 全一
- INIT_TYPE_RAND: 随机
- INIT_TYPE_FILE: 从文件中读取，并进行缩放
- INIT_TYPE_FILE_DISTURB: 从文件中读取，并进行缩放，然后进行扰动
- INIT_TYPE_FILE_ZERO_COMBINE: 从文件中读取，并进行缩放，然后和全零向量进行组合
- INIT_TYPE_FILE_RAND_COMBINE: 从文件中读取，并进行缩放，然后和随机向量进行组合
"""
@enum InitType begin
    INIT_TYPE_ZERO
    INIT_TYPE_ONES
    INIT_TYPE_RAND
    INIT_TYPE_FILE
    INIT_TYPE_FILE_DISTURB
    INIT_TYPE_FILE_ZERO_COMBINE
    INIT_TYPE_FILE_RAND_COMBINE
end

"""
读入一个 .txt 文件中的向量
"""
function read_vector(
    file_path::String
)
    file = open(file_path)
    vector = Float64[]
    for line in eachline(file)
        push!(vector, parse(Float64, line))
    end
    close(file)
    return vector
end

"""
读入一个 .txt 文件中的向量，并进行缩放
"""
function read_rescaled_vector(
    scaled_problem::CuScaledQpProblem,
    primal_file::String,
    dual_file::String,
    primal_size::Int64,
    dual_size::Int64
)
    primal_init = read_vector(primal_file)
    dual_init = read_vector(dual_file)

    # rescale
    variable_rescaling = zeros(Float64, primal_size)
    constraint_rescaling = zeros(Float64, dual_size)
    CUDA.copyto!(variable_rescaling, scaled_problem.variable_rescaling)
    CUDA.copyto!(constraint_rescaling, scaled_problem.constraint_rescaling)

    primal_init .=
        primal_init .* variable_rescaling
    dual_init .=
        dual_init .* constraint_rescaling

    return primal_init, dual_init
end

"""
- init_method: 不同初始化的方式
    - INIT_TYPE_ZERO: 全零
    - INIT_TYPE_ONES: 全一
    - INIT_TYPE_RAND: 随机
    - INIT_TYPE_FILE: 从文件中读取，并进行缩放
    - INIT_TYPE_FILE_DISTURB: 从文件中读取，并进行缩放，然后进行扰动
    - INIT_TYPE_FILE_ZERO_COMBINE: 从文件中读取，并进行缩放，然后和全零向量进行组合
    - INIT_TYPE_FILE_RAND_COMBINE: 从文件中读取，并进行缩放，然后和随机向量进行组合
- k_disturb: 扰动的幅度
- given_weight: 文件中读取并缩放所得到的初始解的权重
"""

init_method_num = 1

init_method_seq::Vector{InitType} = append!(
    [
        INIT_TYPE_ZERO,
        INIT_TYPE_RAND,
        INIT_TYPE_FILE,
        INIT_TYPE_FILE_DISTURB
    ],
    append!(
        repeat([INIT_TYPE_FILE_ZERO_COMBINE], 4),
        repeat([INIT_TYPE_FILE_RAND_COMBINE], 4)
    )
)
k_disturb_seq = [0.0, 0.0, 0.0, 1e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
given_weight_seq = [0.0, 0.0, 0.0, 0.0, 0.8, 0.6, 0.4, 0.2, 0.8, 0.6, 0.4, 0.2]

init_method::InitType = init_method_seq[init_method_num]
# 仅 INIT_TYPE_FILE_DISTURB 有效
k_disturb::Float64 = k_disturb_seq[init_method_num]   
# 仅 INIT_TYPE_FILE_ZERO_COMBINE 和 INIT_TYPE_FILE_RAND_COMBINE 有效     
given_weight::Float64 = given_weight_seq[init_method_num]   

"""
初始化原始问题的解 x 和对偶问题的解 y
"""
function init_primal_dual_solution!(
    scaled_problem::CuScaledQpProblem,
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
    primal_size::Int64,
    dual_size::Int64,
    verbosity::Int64
)
    if verbosity >= 3
        Printf.@printf(
            "init_method %s: %s\n",
            rpad(init_method_num, 2),
            rpad(string(init_method), 24)
        )
        println("k_disturb: ", k_disturb)
        println("given_weight: ", given_weight)
        println()
    end

    if init_method == INIT_TYPE_ZERO
        primal_init = zeros(Float64, primal_size)
        dual_init = zeros(Float64, dual_size)
    end

    if init_method == INIT_TYPE_RAND
        primal_init = rand(Float64, primal_size)
        dual_init = rand(Float64, dual_size)
    end

    if init_method == INIT_TYPE_ONES
        primal_init = ones(Float64, primal_size)
        dual_init = ones(Float64, dual_size)
    end

    if init_method in [
        INIT_TYPE_FILE, 
        INIT_TYPE_FILE_DISTURB, 
        INIT_TYPE_FILE_ZERO_COMBINE, 
        INIT_TYPE_FILE_RAND_COMBINE
    ]
        primal_file::String = "tmp/test_solve/test_primal.txt"
        dual_file::String = "tmp/test_solve/test_dual.txt"
        primal_init, dual_init = read_rescaled_vector(
            scaled_problem,
            primal_file,
            dual_file,
            primal_size,
            dual_size
        )

        # 对 x 和 y 进行扰动
        if init_method == INIT_TYPE_FILE_DISTURB 
            primal_init .= max.(
                (
                    primal_init .+ 
                    rand(Float64, primal_size) .* (2 * k_disturb) .- k_disturb
                ),
                0.0
            )
            dual_init .= max.(
                (
                    dual_init .+ 
                    rand(Float64, dual_size) .* (2 * k_disturb) .- k_disturb
                ),
                0.0
            )   
        end

        if init_method == INIT_TYPE_FILE_ZERO_COMBINE
            primal_init .= primal_init .* given_weight
            dual_init .= dual_init .* given_weight
        end

        if init_method == INIT_TYPE_FILE_RAND_COMBINE
            primal_init .= primal_init .* given_weight
            dual_init .= dual_init .* given_weight
            primal_init .+= rand(Float64, primal_size) .* (1 - given_weight)
            dual_init .+= rand(Float64, dual_size) .* (1 - given_weight)
        end

    end

    # copy to GPU
    CUDA.copyto!(current_primal_solution, primal_init)
    CUDA.copyto!(current_dual_solution, dual_init)
end

"""
计算 Kx
"""
function init_primal_product!(
    problem::CuLinearProgrammingProblem,
    current_primal_solution::CuVector{Float64},
    current_primal_product::CuVector{Float64}
)
    # 简单的矩阵向量乘法 product = Kx
    # current_primal_product .= problem.constraint_matrix * current_primal_solution
    CUDA.CUSPARSE.mv!(
        'N', 
        1, 
        problem.constraint_matrix, 
        current_primal_solution, 
        0, 
        current_primal_product, 
        'O', 

        # SpMV（稀疏矩阵-向量乘法）算法，使用 CSR 格式的算法 2
        CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2   
    )
end

"""
计算 Kᵀy
"""
function init_dual_product!(
    problem::CuLinearProgrammingProblem,
    current_dual_solution::CuVector{Float64},
    current_dual_product::CuVector{Float64}
)
    # 简单的矩阵向量乘法 product = Kᵀy
    # current_dual_product .= problem.constraint_matrix_t * current_dual_solution
    CUDA.CUSPARSE.mv!(
        'N', 
        1, 
        problem.constraint_matrix_t, 
        current_dual_solution, 
        0, 
        current_dual_product, 
        'O', 
        CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2
    )
end

"""
Initialize weighted average by initial vector.

- 此处的参数 x, y, Kx, Kᵀy 都是初始的解
- 一定需要进行 copy，否则会出现引用问题
    - 即 solver_state 中的 solution_weighted_avg 中的 x, y, Kx, Kᵀy
    - 会随着 solver_state 中的 x, y, Kx, Kᵀy 变化，从而出错
"""
function initialize_solution_weighted_average(
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
    current_primal_product::CuVector{Float64},
    current_dual_product::CuVector{Float64}
)
    return CuSolutionWeightedAverage(
        copy(current_primal_solution),
        copy(current_dual_solution),
        0,
        0,
        0.0,
        0.0,
        copy(current_primal_product),   # 注意 Kx 的维度和 y 相同
        copy(current_dual_product),     # 注意 Kᵀy 的维度和 x 相同
    )
end

"""
初始化 solver_state

- `x`: current_primal_solution
- `y`: current_dual_solution
- `Δx`: delta_primal
- `Δy`: delta_dual
- `Kx`: current_primal_product
- `Kᵀy`: current_dual_product
"""
function init_state(
    problem::CuLinearProgrammingProblem,
    scaled_problem::CuScaledQpProblem,
    primal_size::Int64,
    dual_size::Int64,
    verbosity::Int64
)
    current_primal_solution = CUDA.zeros(Float64, primal_size)
    current_dual_solution = CUDA.zeros(Float64, dual_size)
    current_primal_product = CUDA.zeros(Float64, dual_size)
    current_dual_product = CUDA.zeros(Float64, primal_size)

    init_primal_dual_solution!(
        scaled_problem,
        current_primal_solution,
        current_dual_solution,
        primal_size,
        dual_size,
        verbosity
    )
    init_primal_product!(
        problem,
        current_primal_solution,
        current_primal_product
    )
    init_dual_product!(
        problem,
        current_dual_solution,
        current_dual_product
    )

    return CuPdhgSolverState(
        current_primal_solution,
        current_dual_solution,
        CUDA.zeros(Float64, primal_size),
        CUDA.zeros(Float64, dual_size),
        current_primal_product,
        current_dual_product,
        initialize_solution_weighted_average(
            current_primal_solution,
            current_dual_solution,
            current_primal_product,
            current_dual_product
        ),
        0.0,                 # η: step_size
        1.0,                 # ω: primal_weight
        false,               # numerical_error
        0.0,                 # cumulative_kkt_passes
        0,                   # total_number_iterations
        nothing,
        nothing,
    )

    # solver_state = CuPdhgSolverState(
    #     CUDA.zeros(Float64, primal_size),    # x: current_primal_solution
    #     CUDA.zeros(Float64, dual_size),      # y: current_dual_solution
    #     CUDA.zeros(Float64, primal_size),    # Δx: delta_primal
    #     CUDA.zeros(Float64, dual_size),      # Δy: delta_dual
    #     CUDA.zeros(Float64, dual_size),      # Kx: current_primal_product
    #     CUDA.zeros(Float64, primal_size),    # Kᵀy: current_dual_product
    #     initialize_solution_weighted_average(primal_size, dual_size),
    #     0.0,                 # η: step_size
    #     1.0,                 # ω: primal_weight
    #     false,               # numerical_error
    #     0.0,                 # cumulative_kkt_passes
    #     0,                   # total_number_iterations
    #     nothing,
    #     nothing,
    # )
    # return solver_state
end