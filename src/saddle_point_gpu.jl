

struct SaddlePointOutput
    """
    The output primal solution vector.
    """
    primal_solution::Vector{Float64}

    """
    The output dual solution vector.
    """
    dual_solution::Vector{Float64}

    """
    One of the possible values from the TerminationReason enum.
    """
    termination_reason::TerminationReason

    """
    Extra information about the termination reason (may be empty).
    """
    termination_string::String

    """
    The total number of algorithmic iterations for the solve.
    """
    iteration_count::Int32

    """
    Detailed statistics about a subset of the iterations. The collection frequency
    is defined by algorithm parameters.
    """
    iteration_stats::Vector{IterationStats}
end


"""
Return the unscaled primal and dual solutions

- 将 x 和 y 恢复到缩放前
"""
function unscaled_saddle_point_output(
    scaled_problem::ScaledQpProblem,
    primal_solution::AbstractVector{Float64},
    dual_solution::AbstractVector{Float64},
    termination_reason::TerminationReason,
    iterations_completed::Int64,
    iteration_stats::Vector{IterationStats},
)
    # Unscale iterates.
    original_primal_solution =
        primal_solution ./ scaled_problem.variable_rescaling
    original_dual_solution = dual_solution ./ scaled_problem.constraint_rescaling
  
    return SaddlePointOutput(
        original_primal_solution,
        original_dual_solution,
        termination_reason,
        termination_reason_to_string(termination_reason),
        iterations_completed,
        iteration_stats,
    )
end

"""
# Params
- `vec`: 需要求norm的CuVector
- `weights`: 权重

# Returns
- `ret`: = sqrt(weights) * norm(vec)
"""
function weighted_norm(
    vec::CuVector{Float64},
    weights::Float64,
)
    tmp = CUDA.norm(vec)
    return sqrt(weights) * tmp
end

"""
带权重的解的信息

- sum_primal_solutions = ∑ η * x
- sum_dual_solutions = ∑ η * y
- sum_primal_solutions_count = ∑ 1
- sum_dual_solutions_count = ∑ 1
- sum_primal_solution_weights = ∑ η
- sum_dual_solution_weights = ∑ η
- sum_primal_product = ∑ η * Kx
- sum_dual_product = ∑ η * Kᵀy
"""
mutable struct CuSolutionWeightedAverage
    sum_primal_solutions::CuVector{Float64}
    sum_dual_solutions::CuVector{Float64}
    sum_primal_solutions_count::Int64
    sum_dual_solutions_count::Int64
    sum_primal_solution_weights::Float64
    sum_dual_solution_weights::Float64
    sum_primal_product::CuVector{Float64}
    sum_dual_product::CuVector{Float64}
end

"""
- `avg_primal_solution` = ∑ η * x / ∑ η
- `avg_dual_solution` = ∑ η * y / ∑ η
- `avg_primal_product` = ∑ η * Kx / ∑ η
- `avg_primal_gradient` = (- ∑ η * Kᵀy / ∑ η) + c
"""
mutable struct CuBufferAvgState
    avg_primal_solution::CuVector{Float64}
    avg_dual_solution::CuVector{Float64}
    avg_primal_product::CuVector{Float64}
    avg_primal_gradient::CuVector{Float64}
end

"""
Initialize weighted average
"""
function initialize_solution_weighted_average(
    primal_size::Int64,
    dual_size::Int64,
)
    return CuSolutionWeightedAverage(
        CUDA.zeros(Float64, primal_size),
        CUDA.zeros(Float64, dual_size),
        0,
        0,
        0.0,
        0.0,
        CUDA.zeros(Float64, dual_size),     # 注意 Kx 的维度和 y 相同
        CUDA.zeros(Float64, primal_size),   # 注意 Kᵀy 的维度和 x 相同
    )
end

"""
Reset weighted average

- 重置所有的加权信息为 0 
    - `sum_primal_solutions = ∑ η * x`
    - `sum_dual_solutions = ∑ η * y`
    - `sum_primal_solutions_count = ∑ 1`
    - `sum_dual_solutions_count = ∑ 1`
    - `sum_primal_solution_weights = ∑ η`
    - `sum_dual_solution_weights = ∑ η`
    - `sum_primal_product = ∑ η * Kx`
    - `sum_dual_product = ∑ η * Kᵀy`
"""
function reset_solution_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
)
    solution_weighted_avg.sum_primal_solutions .=
        CUDA.zeros(Float64, length(solution_weighted_avg.sum_primal_solutions))
    solution_weighted_avg.sum_dual_solutions .=
        CUDA.zeros(Float64, length(solution_weighted_avg.sum_dual_solutions))
    solution_weighted_avg.sum_primal_solutions_count = 0
    solution_weighted_avg.sum_dual_solutions_count = 0
    solution_weighted_avg.sum_primal_solution_weights = 0.0
    solution_weighted_avg.sum_dual_solution_weights = 0.0

    solution_weighted_avg.sum_primal_product .= CUDA.zeros(Float64, length(solution_weighted_avg.sum_dual_solutions))
    solution_weighted_avg.sum_dual_product .= CUDA.zeros(Float64, length(solution_weighted_avg.sum_primal_solutions))
    return
end

"""
Update weighted average of primal solution

- sum_primal_solutions = ∑ η * x
- sum_primal_solutions_count = ∑ 1
- sum_primal_solution_weights = ∑ η
"""
function add_to_primal_solution_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_primal_solution::CuVector{Float64},
    weight::Float64,
)
    @assert solution_weighted_avg.sum_primal_solutions_count >= 0
    solution_weighted_avg.sum_primal_solutions .+=
        current_primal_solution * weight
    solution_weighted_avg.sum_primal_solutions_count += 1
    solution_weighted_avg.sum_primal_solution_weights += weight
    return
end

"""
Update weighted average of dual solution

- sum_dual_solutions = ∑ η * y
- sum_dual_solutions_count = ∑ 1
- sum_dual_solution_weights = ∑ η
"""
function add_to_dual_solution_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_dual_solution::CuVector{Float64},
    weight::Float64,
)
    @assert solution_weighted_avg.sum_dual_solutions_count >= 0
    solution_weighted_avg.sum_dual_solutions .+= current_dual_solution * weight
    solution_weighted_avg.sum_dual_solutions_count += 1
    solution_weighted_avg.sum_dual_solution_weights += weight
    return
end

"""
Update weighted average of primal product

- sum_primal_product = ∑ η * Kx
"""
function add_to_primal_product_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_primal_product::CuVector{Float64},
    weight::Float64,
)
    @assert solution_weighted_avg.sum_primal_solutions_count >= 0
    solution_weighted_avg.sum_primal_product .+=
        current_primal_product * weight
    return
end

"""
Update weighted average of dual product

- sum_dual_product = ∑ η * K'y
"""
function add_to_dual_product_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_dual_product::CuVector{Float64},
    weight::Float64,
)
    @assert solution_weighted_avg.sum_dual_solutions_count >= 0
    solution_weighted_avg.sum_dual_product .+=
        current_dual_product * weight
    return
end


"""
Update weighted average

- 更新 primal / dual solution 的加权信息
- 说明:
    - sum_primal_solutions = ∑ η * x
    - sum_dual_solutions = ∑ η * y
    - sum_primal_solutions_count = ∑ 1
    - sum_dual_solutions_count = ∑ 1    
    - sum_primal_solution_weights = ∑ η
    - sum_dual_solution_weights = ∑ η
    - sum_primal_product = ∑ η * Kx
    - sum_dual_product = ∑ η * K'y
"""
function add_to_solution_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
    weight::Float64,
    current_primal_product::CuVector{Float64},
    current_dual_product::CuVector{Float64},
)
    add_to_primal_solution_weighted_average!(
        solution_weighted_avg,
        current_primal_solution,
        weight,
    )
    add_to_dual_solution_weighted_average!(
        solution_weighted_avg,
        current_dual_solution,
        weight,
    )

    add_to_primal_product_weighted_average!(
        solution_weighted_avg,
        current_primal_product,
        weight,
    )
    add_to_dual_product_weighted_average!(
        solution_weighted_avg,
        current_dual_product,
        weight,
    )
    return
end

"""
Compute average solutions

- `avg_primal_solution` = ∑ η * x / ∑ η
- `avg_dual_solution` = ∑ η * y / ∑ η
- `avg_primal_product` = ∑ η * Kx / ∑ η
- `avg_primal_gradient` = c - ∑ η * Kᵀy / ∑ η
"""
function compute_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    buffer_avg::CuBufferAvgState,
    problem::CuLinearProgrammingProblem,
)
    buffer_avg.avg_primal_solution .= (
        solution_weighted_avg.sum_primal_solutions 
        ./ solution_weighted_avg.sum_primal_solution_weights
    )
    buffer_avg.avg_dual_solution .= (
        solution_weighted_avg.sum_dual_solutions 
        ./ solution_weighted_avg.sum_dual_solution_weights
    )
    buffer_avg.avg_primal_product .= (
        solution_weighted_avg.sum_primal_product 
        ./ solution_weighted_avg.sum_primal_solution_weights
    )
    buffer_avg.avg_primal_gradient .= (
        -solution_weighted_avg.sum_dual_product 
        ./ solution_weighted_avg.sum_dual_solution_weights
    )
    buffer_avg.avg_primal_gradient .+= problem.objective_vector

end


mutable struct CuKKTrestart
    kkt_residual::Float64
end

"""
Compute weighted KKT residual for restarting

- 计算 KKT_w(z) =
    - `ω² * ||(Ax-b, [h-Gx]+)||₂²`  *?*
    - `+ 1/ω² * ||c-Kᵀy-λ||₂²`      *?*
    - `+ |qᵀy + lᵀλ⁺ - uᵀλ⁻ - cᵀx|²`
"""
function compute_weight_kkt_residual(
    problem::CuLinearProgrammingProblem,
    primal_iterate::CuVector{Float64},  # x
    dual_iterate::CuVector{Float64},    # y
    primal_product::CuVector{Float64},  # Kx
    primal_gradient::CuVector{Float64}, # Kᵀy
    buffer_kkt::BufferKKTState,
    primal_weight::Float64,             # ω
    primal_norm_params::Float64,        # 1 / η * ω
    dual_norm_params::Float64,          # 1 / η / ω
)
    ### construct buffer_kkt
    buffer_kkt.primal_solution .= copy(primal_iterate)
    buffer_kkt.dual_solution .= copy(dual_iterate)
    buffer_kkt.primal_product .= copy(primal_product)
    buffer_kkt.primal_gradient .= copy(primal_gradient)

    # 计算原始问题的解的上界和下界的违反向量
    # 计算等式和不等式约束的违反向量
    compute_primal_residual!(problem, buffer_kkt)

    # 计算原始问题的目标函数值cᵀx + constant
    primal_objective = primal_obj(problem, buffer_kkt.primal_solution)

    # 计算原始问题的三个违反向量的 l2 范数
    # ||(Ax-b, [h-Gx]+)||₂²
    l2_primal_residual = CUDA.norm(
        [
            buffer_kkt.constraint_violation; 
            buffer_kkt.lower_variable_violation; 
            buffer_kkt.upper_variable_violation
        ], 
        2
    )

    # 计算对偶违反向量和对偶obj
    compute_dual_stats!(problem, buffer_kkt)
    dual_objective = buffer_kkt.dual_stats.dual_objective

    # 计算对偶违反向量的 l2 范数
    # ||c-Kᵀy-λ||₂²
    l2_dual_residual = CUDA.norm(
        [
            buffer_kkt.dual_stats.dual_residual; 
            buffer_kkt.reduced_costs_violation
        ],
        2
    )

    # 计算 KKT_w(z)
    weighted_kkt_residual = sqrt(
        (primal_weight * l2_primal_residual^2)      # ? ω² * ||(Ax-b, [h-Gx]+)||₂²
        + (1/primal_weight * l2_dual_residual^2)    # ? 1/ω² * ||c-Kᵀy-λ||₂²
        + abs(primal_objective - dual_objective)^2  # |qᵀy + lᵀλ⁺ - uᵀλ⁻ - cᵀx|²
    )

    return CuKKTrestart(weighted_kkt_residual)
end

mutable struct CuRestartInfo
    """
    The primal_solution recorded at the last restart point.
    """
    primal_solution::CuVector{Float64}
    """
    The dual_solution recorded at the last restart point.
    """
    dual_solution::CuVector{Float64}
    """
    KKT residual at last restart. This has a value of nothing 
    if no restart has occurred.
    """
    last_restart_kkt_residual::Union{Nothing,CuKKTrestart} 
    """
    The length of the last restart interval.
    """
    last_restart_length::Int64
    """
    The primal distance moved from the restart point two restarts 
    ago and the average of the iterates across the last restart.
    """
    primal_distance_moved_last_restart_period::Float64
    """
    The dual distance moved from the restart point two restarts 
    ago and the average of the iterates across the last restart.
    """
    dual_distance_moved_last_restart_period::Float64
    """
    Reduction in the potential function that was achieved last 
    time we tried to do a restart.
    """
    kkt_reduction_ratio_last_trial::Float64

    primal_product::CuVector{Float64}
    primal_gradient::CuVector{Float64}
end

"""
Initialize last restart info
"""
function create_last_restart_info(
    problem::CuLinearProgrammingProblem,
    primal_solution::CuVector{Float64},
    dual_solution::CuVector{Float64},
    primal_product::CuVector{Float64},
    primal_gradient::CuVector{Float64},
)
    return CuRestartInfo(
        copy(primal_solution),
        copy(dual_solution),
        nothing,
        1,
        0.0,
        0.0,
        1.0,
        copy(primal_product),
        copy(primal_gradient),
    )
end

"""
-  `NO_RESTARTS`: No restarts are performed.
-  `FIXED_FREQUENCY`: does a restart every [restart_frequency] iterations where [restart_frequency] is a user-specified number.
-  `ADAPTIVE_KKT`: a heuristic based on the KKT residual to decide when to restart. 
"""
@enum RestartScheme NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT

"""
- `NO_RESTART_TO_CURRENT`: Always reset to the average.
- `KKT_GREEDY`: Decide between the average current based on which has a smaller KKT.
"""
@enum RestartToCurrentMetric NO_RESTART_TO_CURRENT KKT_GREEDY

mutable struct RestartParameters
    """
    Specifies what type of restart scheme is used.
    """
    restart_scheme::RestartScheme
    """
    Specifies how we decide between restarting to the average or current.
    """
    restart_to_current_metric::RestartToCurrentMetric
    """
    If `restart_scheme` = `FIXED_FREQUENCY` then this number determines the frequency 
    that the algorithm is restarted.
    """
    restart_frequency_if_fixed::Int64
    """
    If in the past `artificial_restart_threshold` fraction of iterations no restart 
    has occurred then a restart will be artificially triggered. The value should be 
    between zero and one. Smaller values will have more frequent artificial restarts 
    than larger values.
    """
    artificial_restart_threshold::Float64
    """
    Only applies when `restart_scheme` = `ADAPTIVE`. It is the threshold improvement 
    in the quality of the current/average iterate compared with that  of the last restart 
    that will trigger a restart. The value of this parameter should be between zero and one. 
    Smaller values make restarts less frequent, larger values make restarts more frequent.
    """
    sufficient_reduction_for_restart::Float64
    """
    Only applies when `restart_scheme` = `ADAPTIVE`. It is the threshold
    improvement in the quality of the current/average iterate compared with that of the 
    last restart that is neccessary for a restart to be triggered. If this thrshold is met 
    and the quality of the iterates appear to be getting worse then a restart is triggered. 
    The value of this parameter should be between zero and one, and greater than 
    sufficient_reduction_for_restart. Smaller values make restarts less frequent, 
    larger values make restarts more frequent.
    """
    necessary_reduction_for_restart::Float64
    """
    # * 每次迭代求 ω^n 时的smoothing操作
    Controls the exponential smoothing of log(primal_weight) when the primal weight 
    is updated (i.e., on every restart). Must be between 0.0 and 1.0 inclusive. 
    At 0.0 the primal weight remains frozen at its initial value.
    """
    primal_weight_update_smoothing::Float64
end

"""
Construct restart parameters
"""
function construct_restart_parameters(
    restart_scheme::RestartScheme,
    restart_to_current_metric::RestartToCurrentMetric,
    restart_frequency_if_fixed::Int64,
    artificial_restart_threshold::Float64,
    sufficient_reduction_for_restart::Float64,
    necessary_reduction_for_restart::Float64,
    primal_weight_update_smoothing::Float64,
)
    @assert restart_frequency_if_fixed > 1
    @assert 0.0 < artificial_restart_threshold <= 1.0
    @assert 0.0 <
            sufficient_reduction_for_restart <=
            necessary_reduction_for_restart <=
            1.0
    @assert 0.0 <= primal_weight_update_smoothing <= 1.0
  
    return RestartParameters(
        restart_scheme,
        restart_to_current_metric,
        restart_frequency_if_fixed,
        artificial_restart_threshold,
        sufficient_reduction_for_restart,
        necessary_reduction_for_restart,
        primal_weight_update_smoothing,
    )
end

"""
Check if restart at average solutions

- 如果是 `KKT_GREEDY`
    - 如果平均KKT较小，就重置为平均解
    - 否则重置为当前解
- 如果是 `NO_RESTART_TO_CURRENT`
    - 始终重置为平均解
"""
function should_reset_to_average(
    current::CuKKTrestart,
    average::CuKKTrestart,
    restart_to_current_metric::RestartToCurrentMetric,
)
    # Greedy，如果平均KKT较小，就重置为平均解，否则重置为当前解
    if restart_to_current_metric == KKT_GREEDY
        return current.kkt_residual  >=  average.kkt_residual
    else
        return true # reset to average
    end
end

"""
Check restart criteria based on weighted KKT
"""
function should_do_adaptive_restart_kkt(
    problem::CuLinearProgrammingProblem,
    candidate_kkt::CuKKTrestart, 
    restart_params::RestartParameters,
    last_restart_info::CuRestartInfo,
    primal_weight::Float64,
    buffer_kkt::BufferKKTState,
    primal_norm_params::Float64,
    dual_norm_params::Float64,
)
    
    # 计算上一次 restart 的 KKT
    last_restart = compute_weight_kkt_residual(
        problem,
        last_restart_info.primal_solution,
        last_restart_info.dual_solution,
        last_restart_info.primal_product,
        last_restart_info.primal_gradient,
        buffer_kkt,
        primal_weight,
        primal_norm_params,
        dual_norm_params,
    )

    do_restart = false

    # candidate KKT 值
    kkt_candidate_residual = candidate_kkt.kkt_residual
    # 上一次的 KKT 值
    kkt_last_residual = last_restart.kkt_residual    
    # ratio = candidate KKT / last KKT  
    kkt_reduction_ratio = kkt_candidate_residual / kkt_last_residual

    # ratio < 0.8 = β_necessary
    if kkt_reduction_ratio < restart_params.necessary_reduction_for_restart

        # ratio < 0.2 = β_sufficient
        if kkt_reduction_ratio < restart_params.sufficient_reduction_for_restart
            do_restart = true

        # ratio > last ratio
        elseif kkt_reduction_ratio > last_restart_info.kkt_reduction_ratio_last_trial
            do_restart = true
        end
    end

    # 记录当前 ratio 到 last ratio 中
    last_restart_info.kkt_reduction_ratio_last_trial = kkt_reduction_ratio
  
    return do_restart
end


"""
Check restart

- 检查是否需要 restart
- 进行 restart
- 更新 last_restart_info
- 返回 restart 方案
"""
function run_restart_scheme(
    problem::CuLinearProgrammingProblem,
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
    last_restart_info::CuRestartInfo,
    iterations_completed::Int64,    # k (total iteration number)
    primal_norm_params::Float64,    # 1 / η * ω
    dual_norm_params::Float64,      # 1 / η / ω
    primal_weight::Float64,         # ω
    verbosity::Int64,
    restart_params::RestartParameters,
    primal_product::CuVector{Float64},  
    dual_product::CuVector{Float64},
    buffer_avg::CuBufferAvgState,
    buffer_kkt::BufferKKTState,
    buffer_primal_gradient::CuVector{Float64},
)
    # 还没有迭代过
    if solution_weighted_avg.sum_primal_solutions_count > 0 &&
        solution_weighted_avg.sum_dual_solutions_count > 0
        # compute_average!(solution_weighted_avg, buffer_avg, problem)
    else
        return RESTART_CHOICE_NO_RESTART
    end

    # restart长度
    restart_length = solution_weighted_avg.sum_primal_solutions_count
    # 是否人为触发restart
    artificial_restart = false
    # 是否restart
    do_restart = false
    
    # long inner loop 
    # t ≥ β_artificial * k = 0.36k
    if restart_length >= restart_params.artificial_restart_threshold * iterations_completed
        do_restart = true
        artificial_restart = true
    end

    # restart方案为不进行restart
    if restart_params.restart_scheme == NO_RESTARTS
        reset_to_average = false
        candidate_kkt_residual = nothing

    # restart方案为FIXED_FREQUENCY或者ADAPTIVE_KKT
    else

        # 计算当前的 KKT 值
        current_kkt_res = compute_weight_kkt_residual(
            problem,
            current_primal_solution,
            current_dual_solution,
            primal_product,
            buffer_primal_gradient,
            buffer_kkt,
            primal_weight,
            primal_norm_params,
            dual_norm_params,
        )

        # 计算平均的 KKT 值
        avg_kkt_res = compute_weight_kkt_residual(
            problem,
            buffer_avg.avg_primal_solution,
            buffer_avg.avg_dual_solution,
            buffer_avg.avg_primal_product,
            buffer_avg.avg_primal_gradient,
            buffer_kkt,
            primal_weight,
            primal_norm_params,
            dual_norm_params,
        )

        # 选择 Restart Candidate
        # 判断 candidate 是当前解还是平均值
        reset_to_average = should_reset_to_average(
            current_kkt_res,
            avg_kkt_res,
            restart_params.restart_to_current_metric,
        )
        if reset_to_average     # 平均值
            candidate_kkt_residual = avg_kkt_res
        else    # 当前解
            candidate_kkt_residual = current_kkt_res
        end
    end

    # 如果仍然不需要restart，进一步判断
    if !do_restart
        # Decide if we are going to do a restart.

        # ADAPTIVE_KKT
        if restart_params.restart_scheme == ADAPTIVE_KKT

            # 根据 Restart Criteria 的条件 (1)(2) 来判断是否需要 restart
            do_restart = should_do_adaptive_restart_kkt(
                problem,
                candidate_kkt_residual,
                restart_params,
                last_restart_info,
                primal_weight,
                buffer_kkt,
                primal_norm_params,
                dual_norm_params,
            )

        # FIXED_FREQUENCY
        elseif restart_params.restart_scheme == FIXED_FREQUENCY &&
            restart_params.restart_frequency_if_fixed <= restart_length
            do_restart = true
        end
    end

    if !do_restart
        return RESTART_CHOICE_NO_RESTART
    else
        
        # 如果需要 Restart

        # 是否需要重置为平均值
        if reset_to_average
            if verbosity >= 4
                print("  Restarted to average")
            end
            # 重置为平均值
            # x
            current_primal_solution .= copy(buffer_avg.avg_primal_solution)
            # y
            current_dual_solution .= copy(buffer_avg.avg_dual_solution)
            # Kx
            primal_product .= copy(buffer_avg.avg_primal_product)
            # Kᵀy = c - (c - Kᵀy)
            dual_product .= problem.objective_vector .- buffer_avg.avg_primal_gradient
            # c - Kᵀy
            buffer_primal_gradient .= copy(buffer_avg.avg_primal_gradient)
        else
            # Current point is much better than average point.
            if verbosity >= 4
                print("  Restarted to current")
            end
        end

        if verbosity >= 4
            print(" after ", rpad(restart_length, 4), " iterations")
            if artificial_restart
                println("*")
            else
                println("")
            end
        end

        # 重置加权平均值为 0
        reset_solution_weighted_average!(solution_weighted_avg)

        # 将当前的迭代信息记录到 last_restart_info 中
        update_last_restart_info!(
            last_restart_info,
            current_primal_solution,
            current_dual_solution,
            buffer_avg.avg_primal_solution,
            buffer_avg.avg_dual_solution,
            primal_weight,
            primal_norm_params,
            dual_norm_params,
            candidate_kkt_residual,
            restart_length,
            primal_product,
            buffer_primal_gradient,
        )

        if reset_to_average
            # The algorithm is restarted at the average of iterates since the last restart.
            return RESTART_CHOICE_RESTART_TO_AVERAGE
        else
            # The weighted average of iterates is cleared and reset to the current point.
            return RESTART_CHOICE_WEIGHTED_AVERAGE_RESET
        end
    end
end

"""
Compute primal weight at restart

- 根据以前的 ω 计算新的 ωⁿ
- 如果 Δxⁿ, Δyⁿ > eps()
    - ωⁿ = exp(θ * log(Δyⁿ / Δxⁿ) + (1 - θ) * log(ω))
- 否则
    - ωⁿ = ω
"""
function compute_new_primal_weight(
    last_restart_info::CuRestartInfo,
    primal_weight::Float64,
    primal_weight_update_smoothing::Float64,
    verbosity::Int64,
)
    primal_distance = last_restart_info.primal_distance_moved_last_restart_period
    dual_distance = last_restart_info.dual_distance_moved_last_restart_period
    
    if primal_distance > eps() && dual_distance > eps()
        new_primal_weight_estimate = dual_distance / primal_distance
        # Exponential moving average.
        # If primal_weight_update_smoothing = 1.0 then there is no smoothing.
        # If primal_weight_update_smoothing = 0.0 then the primal_weight is frozen.
        log_primal_weight =
            primal_weight_update_smoothing * log(new_primal_weight_estimate) +
            (1 - primal_weight_update_smoothing) * log(primal_weight)

        primal_weight = exp(log_primal_weight)
        if verbosity >= 4
            Printf.@printf "  New computed primal weight is %.2e\n" primal_weight
        end

        return primal_weight
    else
        return primal_weight
    end
end

"""
Update last restart info

- 将当前的迭代信息记录到 last_restart_info 中
"""
function update_last_restart_info!(
    last_restart_info::CuRestartInfo,
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
    avg_primal_solution::CuVector{Float64},
    avg_dual_solution::CuVector{Float64},
    primal_weight::Float64,
    primal_norm_params::Float64,
    dual_norm_params::Float64,
    candidate_kkt_residual::Union{Nothing,CuKKTrestart},
    restart_length::Int64,
    primal_product::CuVector{Float64},
    primal_gradient::CuVector{Float64},
)
    last_restart_info.primal_distance_moved_last_restart_period =
        weighted_norm(
            avg_primal_solution - last_restart_info.primal_solution,
            primal_norm_params,     # 1 / η * ω
        ) / sqrt(primal_weight)
    last_restart_info.dual_distance_moved_last_restart_period =
        weighted_norm(
            avg_dual_solution - last_restart_info.dual_solution,
            dual_norm_params,       # 1 / η / ω
        ) * sqrt(primal_weight)
    last_restart_info.primal_solution .= copy(current_primal_solution)
    last_restart_info.dual_solution .= copy(current_dual_solution)

    last_restart_info.last_restart_length = restart_length
    last_restart_info.last_restart_kkt_residual = candidate_kkt_residual

    last_restart_info.primal_product .= copy(primal_product)
    last_restart_info.primal_gradient .= copy(primal_gradient)

end

"
根据point_type的值，打印相应的字符串
- `POINT_TYPE_CURRENT_ITERATE`: \"current\"
- `POINT_TYPE_AVERAGE_ITERATE`: \"average\"
- `POINT_TYPE_ITERATE_DIFFERENCE`: \"difference\"
- `other`: \"unknown PointType\"
"
function point_type_label(point_type::PointType)
    if point_type == POINT_TYPE_CURRENT_ITERATE
        return "current"
    elseif point_type == POINT_TYPE_AVERAGE_ITERATE
        return "average"
    elseif point_type == POINT_TYPE_ITERATE_DIFFERENCE
        return "difference"
    else
        return "unknown PointType"
    end
end

"""
- 打印其他信息
"""
function generic_final_log(
    problem::QuadraticProgrammingProblem,
    current_primal_solution::Vector{Float64},
    current_dual_solution::Vector{Float64},
    last_iteration_stats::IterationStats,
    verbosity::Int64,
    iteration::Int64,
    termination_reason::TerminationReason,
)
    # verbosity >= 1 代表迭代终止
    if verbosity >= 1
        print("Terminated after $iteration iterations: ")
        println(termination_reason_to_string(termination_reason))
    end

    method_specific_stats = last_iteration_stats.method_specific_stats

    # verbosity >= 3 代表打印最后一次迭代的信息
    if verbosity >= 3
        for convergence_information in last_iteration_stats.convergence_information
            Printf.@printf(
                "For %s candidate:\n",
                point_type_label(convergence_information.candidate_type)
            )
            # Print more decimal places for the primal and dual objective.
            Printf.@printf(
                "Primal objective: %f, ",
                convergence_information.primal_objective
            )
            Printf.@printf(
                "dual objective: %f, ",
                convergence_information.dual_objective
            )
            Printf.@printf(
                "corrected dual objective: %f \n",
                convergence_information.corrected_dual_objective
            )
        end
    end

    # verbosity >= 4 代表打印两个时间信息 Basic algorithm 和 Full algorithm
    if verbosity >= 4
        Printf.@printf(
            "Time (seconds):\n - Basic algorithm: %.2e\n - Full algorithm:  %.2e\n",
            method_specific_stats["time_spent_doing_basic_algorithm"],
            last_iteration_stats.cumulative_time_sec,
        )
    end

    # verbosity >= 7 代表打印结果的残差范数
    if verbosity >= 7
        for convergence_information in last_iteration_stats.convergence_information
            print_infinity_norms(convergence_information)
        end
        print_variable_and_constraint_hardness(
            problem,
            current_primal_solution,
            current_dual_solution,
        )
    end
end

"""
Initialize primal weight

- `ω`: primal_weight, ω = 1.0 * (||c||_2 / ||b||_2)
- `c`: OBJ, 即目标向量
- `q`: RHS, 即右侧向量（或者`b`）
"""
function select_initial_primal_weight(
    problem::CuLinearProgrammingProblem,
    primal_norm_params::Float64,
    dual_norm_params::Float64,
    primal_importance::Float64,
    verbosity::Int64,
)
    # RHS的加权二范数（此处权重为1.0）
    rhs_vec_norm = weighted_norm(problem.right_hand_side, dual_norm_params)
    # OBJ的加权二范数（此处权重为1.0）
    obj_vec_norm = weighted_norm(problem.objective_vector, primal_norm_params)

    # primal_importance = 1.0
    if obj_vec_norm > 0.0 && rhs_vec_norm > 0.0
        primal_weight = primal_importance * (obj_vec_norm / rhs_vec_norm)
    else
        primal_weight = primal_importance
    end
    if verbosity >= 6
        println("Initial primal weight = $primal_weight")
    end
    return primal_weight
end

