"""
- 自适应步长
    - η' ← min( [1-(k+1)^(-0.3)η̄ ], [(1+(k+1)^(-0.6))η] )
- 参数说明
    - reduction_exponent = 0.3
    - growth_exponent = 0.6
"""
struct AdaptiveStepsizeParams
    reduction_exponent::Float64
    growth_exponent::Float64
end

"""
- 固定步长
"""
struct ConstantStepsizeParams end

"""
- PDHG参数
"""
struct PdhgParameters
    l_inf_ruiz_iterations::Int
    l2_norm_rescaling::Bool

    # * Pock Chambolle 算法
    pock_chambolle_alpha::Union{Float64,Nothing}

    primal_importance::Float64
    scale_invariant_initial_primal_weight::Bool
    verbosity::Int64
    record_iteration_stats::Bool
    termination_evaluation_frequency::Int32
    termination_criteria::TerminationCriteria
    restart_params::RestartParameters
    step_size_policy_params::Union{
        AdaptiveStepsizeParams,
        ConstantStepsizeParams,
    }
end

"""
- CuPDHG 求解器状态
- 用于存储当前的 primal / dual solution
- 参数说明:
    - `x`: current_primal_solution
    - `y`: current_dual_solution
    - `Δx`: delta_primal
    - `Δy`: delta_dual
    - `Kx`: current_primal_product
    - `Kᵀy`: current_dual_product
    - `η`: step_size
    - `ω`: primal_weight
"""
mutable struct CuPdhgSolverState
    current_primal_solution::CuVector{Float64}
    current_dual_solution::CuVector{Float64}
    delta_primal::CuVector{Float64}
    delta_dual::CuVector{Float64}
    current_primal_product::CuVector{Float64}
    current_dual_product::CuVector{Float64}
    solution_weighted_avg::CuSolutionWeightedAverage 
    step_size::Float64
    primal_weight::Float64
    numerical_error::Bool
    cumulative_kkt_passes::Float64
    total_number_iterations::Int64
    required_ratio::Union{Float64,Nothing}
    ratio_step_sizes::Union{Float64,Nothing}
end

"""
- CuPDHG 缓存状态
- 用于存储下一次迭代的 primal / dual solution
- 参数说明:
    - `x'`: next_primal
    - `y'`: next_dual
    - `Δx`: delta_primal
    - `Δy`: delta_dual
    - `Kx'`: next_primal_product
    - `Kᵀy'`: next_dual_product
    - `Δ(Kᵀy)`: delta_dual_product
"""
mutable struct CuBufferState
    next_primal::CuVector{Float64}
    next_dual::CuVector{Float64}
    delta_primal::CuVector{Float64}
    delta_dual::CuVector{Float64}
    next_primal_product::CuVector{Float64}
    next_dual_product::CuVector{Float64}
    delta_dual_product::CuVector{Float64}
end

"""
- 计算 norms

# Returns
- `1 / η * ω`: primal_norm
- `1 / η / ω`: dual_norm
"""
function define_norms(
    primal_size::Int64,
    dual_size::Int64,
    step_size::Float64,
    primal_weight::Float64,
)
    return 1 / step_size * primal_weight, 1 / step_size / primal_weight
end

"""
打印当前求解的信息
"""
function pdhg_specific_log(
    # problem::QuadraticProgrammingProblem,
    iteration::Int64,
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
    step_size::Float64,
    required_ratio::Union{Float64,Nothing},
    primal_weight::Float64,
)
    Printf.@printf(
        # "   %5d inv_step_size=%9g ",
        "   %5d norms=(%9g, %9g) inv_step_size=%9g ",
        iteration,
        CUDA.norm(current_primal_solution),
        CUDA.norm(current_dual_solution),
        1 / step_size,
    )
    if !isnothing(required_ratio)
        Printf.@printf(
        "   primal_weight=%18g  inverse_ss=%18g\n",
        primal_weight,
        required_ratio
        )
    else
        Printf.@printf(
        "   primal_weight=%18g \n",
        primal_weight,
        )
    end
end

"""
打印最终求解的信息
"""
function pdhg_final_log(
    problem::QuadraticProgrammingProblem,
    avg_primal_solution::Vector{Float64},
    avg_dual_solution::Vector{Float64},
    verbosity::Int64,
    iteration::Int64,
    termination_reason::TerminationReason,
    last_iteration_stats::IterationStats,
)

    if verbosity >= 2
        # infeas = max_primal_violation(problem, avg_primal_solution)
        # primal_obj_val = primal_obj(problem, avg_primal_solution)
        # dual_stats =
        #     compute_dual_stats(problem, avg_primal_solution, avg_dual_solution)
        
        println("Avg solution:")
        Printf.@printf(
            "  pr_infeas=%12g pr_obj=%15.10g dual_infeas=%12g dual_obj=%15.10g\n",
            last_iteration_stats.convergence_information[1].l_inf_primal_residual,
            last_iteration_stats.convergence_information[1].primal_objective,
            last_iteration_stats.convergence_information[1].l_inf_dual_residual,
            last_iteration_stats.convergence_information[1].dual_objective
        )
        # primal solution 的范数
        Printf.@printf(
            "  primal norms: L1=%15.10g, L2=%15.10g, Linf=%15.10g\n",
            CUDA.norm(avg_primal_solution, 1),
            CUDA.norm(avg_primal_solution),
            CUDA.norm(avg_primal_solution, Inf)
        )
        # dual solution 的范数
        Printf.@printf(
            "  dual norms:   L1=%15.10g, L2=%15.10g, Linf=%15.10g\n",
            CUDA.norm(avg_dual_solution, 1),
            CUDA.norm(avg_dual_solution),
            CUDA.norm(avg_dual_solution, Inf)
        )
    end

    # 打印其他信息
    generic_final_log(
        problem,
        avg_primal_solution,
        avg_dual_solution,
        last_iteration_stats,
        verbosity,
        iteration,
        termination_reason,
    )
end

"""
Power method 失败概率
"""
function power_method_failure_probability(
    dimension::Int64,
    epsilon::Float64,
    k::Int64,
)
    if k < 2 || epsilon <= 0.0
        return 1.0
    end
    return (
        min(0.824, 0.354 / sqrt(epsilon * (k - 1))) 
        * sqrt(dimension) 
        * (1.0 - epsilon)^(k - 1 / 2)
    ) # FirstOrderLp.jl old version (epsilon * (k - 1)) instead of sqrt(epsilon * (k - 1)))
end

"""
Power method 估计最大奇异值
奇异值 = sqrt(λ_max)，其中 λ_max 为 A^T * A 的最大特征值
"""
function estimate_maximum_singular_value(
    matrix::SparseMatrixCSC{Float64,Int64};
    probability_of_failure = 0.01::Float64,
    desired_relative_error = 0.1::Float64,
    seed::Int64 = 1,
)
    epsilon = 1.0 - (1.0 - desired_relative_error)^2

    # 生成随机向量
    x = randn(Random.MersenneTwister(seed), size(matrix, 2))

    # power method 迭代估计主特征值对应的特征向量
    number_of_power_iterations = 0
    while power_method_failure_probability(
        size(matrix, 2),
        epsilon,
        number_of_power_iterations,
    ) > probability_of_failure

        # 归一化
        x = x / norm(x, 2)              

        # power method迭代方法 x^{t+1} = Ax^{t}
        # 此处的 A = matrix' * matrix
        x = matrix' * (matrix * x)          
        number_of_power_iterations += 1
    end
    
    # dot 点积，即各元素乘积之和
    # A = matrix' * matrix
    # 已求得 x 为特征向量
    # 由 Ax = λx 得 Ax⋅x = λx⋅x = λ||x||^2
    # 所以 λ = Ax⋅x / ||x||^2
    # 奇异值 = sqrt(λ)
    return sqrt(dot(x, matrix' * (matrix * x)) / norm(x, 2)^2),
    number_of_power_iterations
end

"""
Kernel to compute primal solution in the next iteration

- `x' ← x - η/ω * (c - Kᵀy)`
- 参数说明
    - `c`: objective_vector
    - `x`: current_primal_solution
    - `Kᵀy`: current_dual_product
    - `η`: step_size
    - `ω`: primal_weight
    - `x'`: next_primal
"""
function compute_next_primal_solution_kernel!(
    objective_vector::CuDeviceVector{Float64},
    variable_lower_bound::CuDeviceVector{Float64},
    variable_upper_bound::CuDeviceVector{Float64},
    current_primal_solution::CuDeviceVector{Float64},
    current_dual_product::CuDeviceVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    num_variables::Int64,
    next_primal::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_variables
        @inbounds begin
            next_primal[tx] = (
                current_primal_solution[tx] 
                - (step_size / primal_weight) 
                * (objective_vector[tx] - current_dual_product[tx])
            )
            next_primal[tx] = min(
                variable_upper_bound[tx], 
                max(
                    variable_lower_bound[tx], 
                    next_primal[tx]
                )
            )
        end
    end
    return 
end

"""
Compute primal solution in the next iteration

- `x' ← x - η/ω * (c - Kᵀy)`
- 参数说明  
    - `c`: objective_vector
    - `x`: current_primal_solution
    - `Kᵀy`: current_dual_product
    - `η`: step_size
    - `ω`: primal_weight
    - `x'`: next_primal
    - `Kx'`: next_primal_product
"""
function compute_next_primal_solution!(
    problem::CuLinearProgrammingProblem,
    current_primal_solution::CuVector{Float64},
    current_dual_product::CuVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    next_primal::CuVector{Float64},
    next_primal_product::CuVector{Float64},
)
    NumBlockPrimal = ceil(Int64, problem.num_variables/ThreadPerBlock)

    # 计算下一次迭代的 primal solution
    CUDA.@sync @cuda threads = 
        ThreadPerBlock blocks = 
        NumBlockPrimal compute_next_primal_solution_kernel!(
            problem.objective_vector,
            problem.variable_lower_bound,
            problem.variable_upper_bound,
            current_primal_solution,
            current_dual_product,
            step_size,
            primal_weight,
            problem.num_variables,
            next_primal,
    )

    # 简单的矩阵向量乘法 product = A * x
    # next_primal_product .= problem.constraint_matrix * next_primal
    CUDA.CUSPARSE.mv!(
        'N', 
        1, 
        problem.constraint_matrix, 
        next_primal, 
        0, 
        next_primal_product, 
        'O', 

        # SpMV（稀疏矩阵-向量乘法）算法，使用 CSR 格式的算法 2
        CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2   
    )
end

"""
Kernel to compute dual solution in the next iteration

- `y' ← y + ηω * (q - Kx' - ρ * (Kx' - Kx))`
- 当 ρ = 1.0 时，为 `y' ← y + ηω * (q - K(2x' - x))`
- 参数说明
    - `q`: right_hand_side
    - `y`: current_dual_solution
    - `Kx`: current_primal_product
    - `Kx'`: next_primal_product
    - `η`: step_size
    - `ω`: primal_weight
    - `ρ`: extrapolation_coefficient (通常为 1.0)
    - `y'`: next_dual
"""
function compute_next_dual_solution_kernel!(
    right_hand_side::CuDeviceVector{Float64},
    current_dual_solution::CuDeviceVector{Float64},
    current_primal_product::CuDeviceVector{Float64},
    next_primal_product::CuDeviceVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    extrapolation_coefficient::Float64,
    num_equalities::Int64,
    num_constraints::Int64,
    next_dual::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_equalities
        @inbounds begin
            next_dual[tx] = ( 
                current_dual_solution[tx] 
                + (primal_weight * step_size) * (
                    right_hand_side[tx] 
                    - next_primal_product[tx] 
                    - extrapolation_coefficient * (
                        next_primal_product[tx] 
                        - current_primal_product[tx]
                    )
                )
            )
        end
    elseif num_equalities + 1 <= tx <= num_constraints
        @inbounds begin
            next_dual[tx] = (
                current_dual_solution[tx] 
                + (primal_weight * step_size) * (
                    right_hand_side[tx] 
                    - next_primal_product[tx] 
                    - extrapolation_coefficient * (
                        next_primal_product[tx] 
                        - current_primal_product[tx]
                    )
                )
            )
            next_dual[tx] = max(next_dual[tx], 0.0)
        end
    end
    return 
end

"""
Compute dual solution in the next iteration

- `y' ← y + ηω * (q - Kx' - ρ * (Kx' - Kx))`
- 当 ρ = 1.0 时，为 `y' ← y + ηω * (q - K(2x' - x))`
- 参数说明
    - `q`: right_hand_side
    - `y`: current_dual_solution
    - `Kx`: current_primal_product
    - `Kx'`: next_primal_product
    - `η`: step_size
    - `ω`: primal_weight
    - `ρ`: extrapolation_coefficient (通常为 1.0)
    - `y'`: next_dual
    - `Kᵀy'`: next_dual_product
"""
function compute_next_dual_solution!(
    problem::CuLinearProgrammingProblem,
    current_dual_solution::CuVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    next_primal_product::CuVector{Float64},
    current_primal_product::CuVector{Float64},
    next_dual::CuVector{Float64},
    next_dual_product::CuVector{Float64};
    extrapolation_coefficient::Float64 = 1.0,
)
    NumBlockDual = ceil(Int64, problem.num_constraints/ThreadPerBlock)

    CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockDual compute_next_dual_solution_kernel!(
        problem.right_hand_side,
        current_dual_solution,
        current_primal_product,
        next_primal_product,
        step_size,
        primal_weight,
        extrapolation_coefficient,
        problem.num_equalities,
        problem.num_constraints,
        next_dual,
    )

    # 简单的矩阵向量乘法 product = A^T * y
    # next_dual_product .= problem.constraint_matrix_t * next_dual
    CUDA.CUSPARSE.mv!(
        'N', 
        1, 
        problem.constraint_matrix_t, 
        next_dual, 
        0, 
        next_dual_product, 
        'O', 
        CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2
    )
end

"""
Update primal and dual solutions

- 更新 primal / dual solution
- 更新 primal / dual solution 的加权信息
    - 如 `∑ η` 等
"""
function update_solution_in_solver_state!(
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
)
    # * Δx = x' - x
    solver_state.delta_primal .= (
        buffer_state.next_primal .- solver_state.current_primal_solution
    )
    # * Δy = y' - y
    solver_state.delta_dual .= (
        buffer_state.next_dual .- solver_state.current_dual_solution
    )
    # solver_state.delta_dual_product .= (
    #    buffer_state.next_dual_product .- solver_state.current_dual_product
    #)

    # * x ← x' 
    solver_state.current_primal_solution .= copy(buffer_state.next_primal)
    # * y ← y'
    solver_state.current_dual_solution .= copy(buffer_state.next_dual)
    # * Kᵀy ← Kᵀy'
    solver_state.current_dual_product .= copy(buffer_state.next_dual_product)
    # * Kx ← Kx'
    solver_state.current_primal_product .= copy(buffer_state.next_primal_product)

    # * η
    weight = solver_state.step_size
    
    # * 更新 primal / dual solution 的加权信息
    add_to_solution_weighted_average!(
        solver_state.solution_weighted_avg,
        solver_state.current_primal_solution,
        solver_state.current_dual_solution,
        weight,
        solver_state.current_primal_product,
        solver_state.current_dual_product,
    )
end

"""
Compute iteraction and movement for AdaptiveStepsize

- `interaction = |Δx ⋅ Δy|`
- `movement = 0.5 * ω * ||Δx||^2 + 0.5/ω * ||Δy||^2`
"""
function compute_interaction_and_movement(
    solver_state::CuPdhgSolverState,
    problem::CuLinearProgrammingProblem,
    buffer_state::CuBufferState,
)
    # * Δx = x' - x
    buffer_state.delta_primal .= (
        buffer_state.next_primal .- solver_state.current_primal_solution
    )

    # * Δy = y' - y
    buffer_state.delta_dual .= (
        buffer_state.next_dual .- solver_state.current_dual_solution
    )

    # * Δ(Kᵀy) = Kᵀy' - Kᵀy
    buffer_state.delta_dual_product .= (
        buffer_state.next_dual_product .- solver_state.current_dual_product
    )

    # * interaction = |Δx ⋅ Δy| 
    # 点积，即各元素乘积之和
    primal_dual_interaction = CUDA.dot(
        buffer_state.delta_primal, 
        buffer_state.delta_dual_product
    ) 
    interaction = abs(primal_dual_interaction) 

    # * norm(Δx) 与 norm(Δy)
    norm_delta_primal = CUDA.norm(buffer_state.delta_primal)
    norm_delta_dual = CUDA.norm(buffer_state.delta_dual)

    # * movement = 0.5 * ω * ||Δx||^2 + 0.5/ω * ||Δy||^2
    movement = (
        0.5 
        * solver_state.primal_weight 
        * norm_delta_primal^2 
        + (0.5 / solver_state.primal_weight) 
        * norm_delta_dual^2
    )
    return interaction, movement
end

"""
Take PDHG step with AdaptiveStepsize

- AdaptiveStepPDHG
- 自适应步长
"""
function take_step!(
    step_params::AdaptiveStepsizeParams,
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
)
    step_size = solver_state.step_size
    done = false

    while !done
        solver_state.total_number_iterations += 1

        # * x' ← x - η/ω * (c - Kᵀy)
        compute_next_primal_solution!(
            problem,
            solver_state.current_primal_solution,
            solver_state.current_dual_product,
            step_size,
            solver_state.primal_weight,
            buffer_state.next_primal,
            buffer_state.next_primal_product,
        )

        # * y' ← y + ηω * (q - Kx' - ρ * (Kx' - Kx))
        compute_next_dual_solution!(
            problem,
            solver_state.current_dual_solution,
            step_size,
            solver_state.primal_weight,
            buffer_state.next_primal_product,
            solver_state.current_primal_product,
            buffer_state.next_dual,
            buffer_state.next_dual_product,
        )

        # * interaction = |Δx ⋅ Δy|
        # * movement = 0.5 * ω * ||Δx||^2 + 0.5/ω * ||Δy||^2
        interaction, movement = compute_interaction_and_movement(
            solver_state,
            problem,
            buffer_state,
        )

        # * cumulative_kkt_passes += 1
        solver_state.cumulative_kkt_passes += 1

        # * 计算 step_size_limit
        if interaction > 0
            step_size_limit = movement / interaction
            if movement == 0.0
                # The algorithm will terminate at the beginning of the next iteration
                solver_state.numerical_error = true
                break
            end
        else
            step_size_limit = Inf
        end

        # * 如果 step_size <= step_size_limit
        # * 则更新 primal / dual solution
        # * 并终止循环，结束这一次迭代
        if step_size <= step_size_limit
            update_solution_in_solver_state!(
                solver_state, 
                buffer_state,
            )
            done = true
        end

        # * (1 - (k + 1) ^ (-0.3))η̄
        first_term = (
            (
                1 - 1/(
                    solver_state.total_number_iterations + 1
                )^(step_params.reduction_exponent)
            ) * step_size_limit
        )

        # * (1 + (k + 1) ^ (-0.6))η
        second_term = (
            (
                1 + 1/(
                    solver_state.total_number_iterations + 1
                )^(step_params.growth_exponent)
            ) * step_size
        )

        # * η' ← min( [1-(k+1)^(-0.3)η̄ ], [(1+(k+1)^(-0.6))η] )
        step_size = min(first_term, second_term)
        
    end  
    # * 更新 step_size
    solver_state.step_size = step_size
end

"""
Take PDHG step with ConstantStepsize

- ConstantStepPDHG
- 固定步长
"""
function take_step!(
    step_params::ConstantStepsizeParams,
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
)
    # x' ← x - η/ω * (c - Kᵀy)
    compute_next_primal_solution!(
        problem,
        solver_state.current_primal_solution,
        solver_state.current_dual_product,
        solver_state.step_size,
        solver_state.primal_weight,
        buffer_state.next_primal,
        buffer_state.next_primal_product,
    )
    
    # y' ← y + ηω * (q - Kx' - ρ * (Kx' - Kx))
    compute_next_dual_solution!(
        problem,
        solver_state.current_dual_solution,
        solver_state.step_size,
        solver_state.primal_weight,
        buffer_state.next_primal_product,
        solver_state.current_primal_product,
        buffer_state.next_dual,
        buffer_state.next_dual_product,
    )

    # cumulative_kkt_passes += 1
    solver_state.cumulative_kkt_passes += 1

    # 更新 primal / dual solution
    update_solution_in_solver_state!(
        solver_state, 
        buffer_state,
    )
end

"""
Main algorithm: given parameters and LP problem, return solutions
"""
function optimize(
    params::PdhgParameters,
    original_problem::QuadraticProgrammingProblem,
    params_other::MIPParameters = MIPParameters()
)
    # <debug>
    debug_reset!()

    # 检查问题参数
    validate(original_problem)

    # 缓存原始问题的信息
    qp_cache = cached_quadratic_program_info(original_problem)

    # 缩放原问题
    scaled_problem = rescale_problem(
        params.l_inf_ruiz_iterations,
        params.l2_norm_rescaling,
        params.pock_chambolle_alpha,
        params.verbosity,
        original_problem,
    )

    # primal_size = n
    primal_size = length(scaled_problem.scaled_qp.variable_lower_bound)
    # dual_size = m
    dual_size = length(scaled_problem.scaled_qp.right_hand_side)
    # num_eq = 等式约束的个数
    num_eq = scaled_problem.scaled_qp.num_equalities
    if params.primal_importance <= 0 || !isfinite(params.primal_importance)
        error("primal_importance must be positive and finite")
    end

    # transfer from cpu to gpu  
    # GPU上的CuScaledQpProblem
    d_scaled_problem = scaledqp_cpu_to_gpu(scaled_problem)
    # GPU上的缩放后的QP问题
    d_problem = d_scaled_problem.scaled_qp
    # GPU上的原始QP问题
    buffer_lp = qp_cpu_to_gpu(original_problem)

    # initialization
    # 求解器状态初始化
    # <extra>
    solver_state = init_state(
        d_problem, 
        d_scaled_problem, 
        primal_size, 
        dual_size,
        params.verbosity
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

    # 状态缓存初始化
    buffer_state = CuBufferState(
        CUDA.zeros(Float64, primal_size),      # x': next_primal
        CUDA.zeros(Float64, dual_size),        # y': next_dual
        CUDA.zeros(Float64, primal_size),      # Δx: delta_primal
        CUDA.zeros(Float64, dual_size),        # Δy: delta_dual
        CUDA.zeros(Float64, dual_size),        # Kx': next_primal_product
        CUDA.zeros(Float64, primal_size),      # Kᵀy': next_dual_product
        CUDA.zeros(Float64, primal_size),      # Δ(Kᵀy): delta_dual_product
    )

    # 平均状态缓存初始化
    buffer_avg = CuBufferAvgState(
        CUDA.zeros(Float64, primal_size),      # ∑ η * x / ∑ η: avg_primal_solution
        CUDA.zeros(Float64, dual_size),        # ∑ η * y / ∑ η: avg_dual_solution
        CUDA.zeros(Float64, dual_size),        # ∑ η * Kx / ∑ η: avg_primal_product
        CUDA.zeros(Float64, primal_size),      # (- ∑ η * Kᵀy / ∑ η) + c: avg_primal_gradient
    )

    # 原始状态缓存初始化
    buffer_original = BufferOriginalSol(
        CUDA.zeros(Float64, primal_size),      # x: primal
        CUDA.zeros(Float64, dual_size),        # y: dual
        CUDA.zeros(Float64, dual_size),        # Kx: primal_product
        CUDA.zeros(Float64, primal_size),      # primal_gradient
    )

    # KKT状态缓存初始化
    buffer_kkt = BufferKKTState(
        CUDA.zeros(Float64, primal_size),      # x: primal
        CUDA.zeros(Float64, dual_size),        # y: dual
        CUDA.zeros(Float64, dual_size),        # Kx: primal_product
        CUDA.zeros(Float64, primal_size),      # primal_gradient
        CUDA.zeros(Float64, primal_size),      # lower_variable_violation
        CUDA.zeros(Float64, primal_size),      # upper_variable_violation
        CUDA.zeros(Float64, dual_size),        # constraint_violation
        CUDA.zeros(Float64, primal_size),      # dual_objective_contribution_array
        CUDA.zeros(Float64, primal_size),      # reduced_costs_violations
        CuDualStats(
            0.0,
            CUDA.zeros(Float64, dual_size - num_eq),
            CUDA.zeros(Float64, primal_size),
        ),
        0.0,                                   # dual_res_inf
    )
    
    # KKT不可行状态缓存初始化
    buffer_kkt_infeas = BufferKKTState(
        CUDA.zeros(Float64, primal_size),      # primal
        CUDA.zeros(Float64, dual_size),        # dual
        CUDA.zeros(Float64, dual_size),        # primal_product
        CUDA.zeros(Float64, primal_size),      # primal_gradient
        CUDA.zeros(Float64, primal_size),      # lower_variable_violation
        CUDA.zeros(Float64, primal_size),      # upper_variable_violation
        CUDA.zeros(Float64, dual_size),        # constraint_violation
        CUDA.zeros(Float64, primal_size),      # dual_objective_contribution_array
        CUDA.zeros(Float64, primal_size),      # reduced_costs_violations
        CuDualStats(
            0.0,
            CUDA.zeros(Float64, dual_size - num_eq),
            CUDA.zeros(Float64, primal_size),
        ),
        0.0,                                   # dual_res_inf
    )

    # 初始化原始问题的梯度 ∇f(x) = c - Kᵀy
    buffer_primal_gradient = CUDA.zeros(Float64, primal_size)
    buffer_primal_gradient .= (
        d_scaled_problem.scaled_qp.objective_vector 
        .- solver_state.current_dual_product
    )

    # stepsize
    # 如果为 AdaptiveStepsizeParams，则初始化 step_size = 1.0 / ||A||_∞
    if params.step_size_policy_params isa AdaptiveStepsizeParams
        solver_state.cumulative_kkt_passes += 0.5
        solver_state.step_size = (
            1.0 / norm(scaled_problem.scaled_qp.constraint_matrix, Inf)
        )

    # 如果为 ConstantStepsizeParams，则初始化 step_size = 0.8 / sqrt( λ_max{AᵀA} )
    else
        desired_relative_error = 0.2
        maximum_singular_value, number_of_power_iterations =
            estimate_maximum_singular_value(
                scaled_problem.scaled_qp.constraint_matrix,
                probability_of_failure = 0.001,
                desired_relative_error = desired_relative_error,
            )
        solver_state.step_size =
            (1 - desired_relative_error) / maximum_singular_value
        solver_state.cumulative_kkt_passes += number_of_power_iterations
    end

    # 每次 evaluation 时，cumulative_kkt_passes 累加的值，为 2.0
    KKT_PASSES_PER_TERMINATION_EVALUATION = 2.0

    # 初始化 primal weight ω = ||c||_2 / ||q||_2
    if params.scale_invariant_initial_primal_weight
        solver_state.primal_weight = select_initial_primal_weight(
            d_scaled_problem.scaled_qp,
            1.0,
            1.0,
            params.primal_importance,
            params.verbosity,
        )
    else
        solver_state.primal_weight = params.primal_importance
    end

    primal_weight_update_smoothing = params.restart_params.primal_weight_update_smoothing 

    # 初始化迭代统计信息
    iteration_stats = IterationStats[]      # 空数组
    start_time = time()
    time_spent_doing_basic_algorithm = 0.0

    # 上一次重启时的信息
    last_restart_info = create_last_restart_info(
        d_scaled_problem.scaled_qp,
        solver_state.current_primal_solution,
        solver_state.current_dual_solution,
        solver_state.current_primal_product,
        buffer_primal_gradient,
    )

    # For termination criteria:
    # 终止条件
    termination_criteria = params.termination_criteria
    # 迭代次数限制
    iteration_limit = termination_criteria.iteration_limit
    # 终止评估频率
    termination_evaluation_frequency = params.termination_evaluation_frequency

    println("ITERATION_LIMIT: ", iteration_limit)

    # This flag represents whether a numerical error occurred during the algorithm
    # if it is set to true it will trigger the algorithm to terminate.
    solver_state.numerical_error = false
    display_iteration_stats_heading(params.verbosity)

    # * Main loop
    iteration = 0
    while true
        iteration += 1

        # 评估终止条件
        if (
            mod(iteration - 1, termination_evaluation_frequency) == 0 ||
            iteration == iteration_limit + 1 ||
            iteration <= 10 ||
            solver_state.numerical_error
        )
            
            # cumulative_kkt_passes += 2.0
            solver_state.cumulative_kkt_passes += KKT_PASSES_PER_TERMINATION_EVALUATION

            ### average ###
            # 出现数值错误，或者 primal / dual solution 为空
            # 则将当前的 x / y / Kx / c-Kᵀy 复制到 buffer_avg 中
            # 否则根据 solver_state 来计算出平均值
            if (
                solver_state.numerical_error 
                || solver_state.solution_weighted_avg.sum_primal_solutions_count == 0 
                || solver_state.solution_weighted_avg.sum_dual_solutions_count == 0
            )
                buffer_avg.avg_primal_solution .= copy(solver_state.current_primal_solution)
                buffer_avg.avg_dual_solution .= copy(solver_state.current_dual_solution)
                buffer_avg.avg_primal_product .= copy(solver_state.current_primal_product)
                buffer_avg.avg_primal_gradient .= copy(buffer_primal_gradient) 
            else
                compute_average!(solver_state.solution_weighted_avg, buffer_avg, d_problem)
            end

            ### KKT ###
            # 恢复到缩放前，并计算迭代信息
            current_iteration_stats = evaluate_unscaled_iteration_stats(
                d_scaled_problem,
                qp_cache,
                params.termination_criteria,
                params.record_iteration_stats,
                buffer_avg.avg_primal_solution,
                buffer_avg.avg_dual_solution,
                iteration,
                time() - start_time,
                solver_state.cumulative_kkt_passes,
                termination_criteria.eps_optimal_absolute,
                termination_criteria.eps_optimal_relative,
                solver_state.step_size,
                solver_state.primal_weight,
                POINT_TYPE_AVERAGE_ITERATE, 
                buffer_avg.avg_primal_product,
                buffer_avg.avg_primal_gradient,
                buffer_original,
                buffer_kkt,
                buffer_kkt_infeas,
                buffer_lp,
            )

            # method_specific_stats::Dict{AbstractString,Float64}
            method_specific_stats = current_iteration_stats.method_specific_stats
            # 记录时间
            method_specific_stats["time_spent_doing_basic_algorithm"] =
                time_spent_doing_basic_algorithm

            # primal norm = 1 / η * ω
            # dual norm = 1 / η / ω
            primal_norm_params, dual_norm_params = define_norms(
                primal_size,
                dual_size,
                solver_state.step_size,
                solver_state.primal_weight,
            )
            
            ### check termination criteria ###
            termination_reason = check_termination_criteria(
                termination_criteria,
                qp_cache,
                current_iteration_stats,
            )

            # 检查是否出现数值错误
            if solver_state.numerical_error && termination_reason == false
                termination_reason = TERMINATION_REASON_NUMERICAL_ERROR
            end

            # If we're terminating, record the iteration stats to provide final
            # solution stats.
            # 将当前的迭代信息记录到 iteration_stats 中
            if params.record_iteration_stats || termination_reason != false
                push!(iteration_stats, current_iteration_stats)
            end

            # Print table.
            # 每 display_frequency 次评估打印一次迭代信息
            if print_to_screen_this_iteration(
                termination_reason,
                iteration,
                params.verbosity,
                termination_evaluation_frequency,
            )
                display_iteration_stats(current_iteration_stats, params.verbosity)
            end

            # 如果迭代终止
            # <extra>
            if termination_reason != false && 
                !(
                    (
                        termination_reason == TERMINATION_REASON_PRIMAL_INFEASIBLE ||
                        termination_reason == TERMINATION_REASON_DUAL_INFEASIBLE
                    ) && (iteration <= 1)
                )
            # if termination_reason != false 
                # ** Terminate the algorithm **
                # This is the only place the algorithm can terminate. 
                # Please keep it this way.
                
                # GPU to CPU (x, y)
                avg_primal_solution = zeros(primal_size)
                avg_dual_solution = zeros(dual_size)
                gpu_to_cpu!(
                    buffer_avg.avg_primal_solution,
                    buffer_avg.avg_dual_solution,
                    avg_primal_solution,
                    avg_dual_solution,
                )

                # 打印最终求解的信息
                pdhg_final_log(
                    scaled_problem.scaled_qp,
                    avg_primal_solution,
                    avg_dual_solution,
                    params.verbosity,
                    iteration,
                    termination_reason,
                    current_iteration_stats,
                )

                # 将 x 和 y 恢复到缩放前
                if params_other.mip_return
                    return unscaled_saddle_point_output(
                        scaled_problem,
                        avg_primal_solution,
                        avg_dual_solution,
                        termination_reason,
                        iteration - 1,
                        IterationStats[iteration_stats[end]],
                    )
                else 
                    return unscaled_saddle_point_output(
                        scaled_problem,
                        avg_primal_solution,
                        avg_dual_solution,
                        termination_reason,
                        iteration - 1,
                        iteration_stats,
                    )
                end
            end

            # primal_gradient = c - Kᵀy
            buffer_primal_gradient .= (
                d_scaled_problem.scaled_qp.objective_vector 
                .- solver_state.current_dual_product
            )

            # 检查是否需要 restart
            # 进行 restart
            current_iteration_stats.restart_used = run_restart_scheme(
                d_scaled_problem.scaled_qp,
                solver_state.solution_weighted_avg,
                solver_state.current_primal_solution,
                solver_state.current_dual_solution,
                last_restart_info,
                iteration - 1,
                primal_norm_params,
                dual_norm_params,
                solver_state.primal_weight,
                params.verbosity,
                params.restart_params,
                solver_state.current_primal_product,
                solver_state.current_dual_product,
                buffer_avg,
                buffer_kkt,
                buffer_primal_gradient,
            )

            # 如果需要 restart
            if current_iteration_stats.restart_used != RESTART_CHOICE_NO_RESTART

                # <debug>
                debug_add!(iteration)

                # 计算新的 ωⁿ (primal_weight)
                solver_state.primal_weight = compute_new_primal_weight(
                    last_restart_info,
                    solver_state.primal_weight,
                    primal_weight_update_smoothing,
                    params.verbosity,
                )
                solver_state.ratio_step_sizes = 1.0
            end
        end

        # 对 take_step! 计时
        time_spent_doing_basic_algorithm_checkpoint = time()
      
        if params.verbosity >= 6 && print_to_screen_this_iteration(
            false, # termination_reason
            iteration,
            params.verbosity,
            termination_evaluation_frequency,
        )
            pdhg_specific_log(
                # problem,
                iteration,
                solver_state.current_primal_solution,
                solver_state.current_dual_solution,
                solver_state.step_size,
                solver_state.required_ratio,
                solver_state.primal_weight,
            )
          end

        take_step!(params.step_size_policy_params, d_problem, solver_state, buffer_state)

        time_spent_doing_basic_algorithm += time() - time_spent_doing_basic_algorithm_checkpoint
    end
end