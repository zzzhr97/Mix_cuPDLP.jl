
"""
- CUDA线性规划问题
- 参数说明:
    - `n`: num_variables
    - `m`: num_constraints
    - `l`: variable_lower_bound
    - `u`: variable_upper_bound
    - `c`: objective_vector
    - `c_0`: objective_constant, 即目标函数的常数项
    - `A`: constraint_matrix
    - `Aᵀ`: constraint_matrix_t
    - `b`: right_hand_side
    - `m_2`: num_equalities, 即等式约束的个数
"""
mutable struct CuLinearProgrammingProblem
    num_variables::Int64
    num_constraints::Int64
    variable_lower_bound::CuVector{Float64}
    variable_upper_bound::CuVector{Float64}
    isfinite_variable_lower_bound::CuVector{Bool}
    isfinite_variable_upper_bound::CuVector{Bool}
    objective_vector::CuVector{Float64}
    objective_constant::Float64
    constraint_matrix::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64,Int32}
    constraint_matrix_t::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64,Int32}
    right_hand_side::CuVector{Float64}
    num_equalities::Int64
end

"""
GPU上的缩放问题

- `original_qp`: 原始问题
- `scaled_qp`: 缩放后的问题
- `constraint_rescaling`: 约束缩放向量，记录了缩放信息
- `variable_rescaling`: 变量缩放向量，记录了缩放信息
"""
mutable struct CuScaledQpProblem
    original_qp::CuLinearProgrammingProblem
    scaled_qp::CuLinearProgrammingProblem
    constraint_rescaling::CuVector{Float64}
    variable_rescaling::CuVector{Float64}
end

"""
Transfer quadratic program from CPU to GPU
"""
function qp_cpu_to_gpu(problem::QuadraticProgrammingProblem)
    num_constraints, num_variables = size(problem.constraint_matrix)
    isfinite_variable_lower_bound = Vector{Bool}(isfinite.(problem.variable_lower_bound))
    isfinite_variable_upper_bound = Vector{Bool}(isfinite.(problem.variable_upper_bound))

    # * l
    d_variable_lower_bound = CuArray{Float64}(undef, num_variables)
    # * u
    d_variable_upper_bound = CuArray{Float64}(undef, num_variables)
    # * l 是否有限
    d_isfinite_variable_lower_bound = CuArray{Bool}(undef, num_variables)
    # * u 是否有限
    d_isfinite_variable_upper_bound = CuArray{Bool}(undef, num_variables)
    # * c
    d_objective_vector = CuArray{Float64}(undef, num_variables)
    # * b
    d_right_hand_side = CuArray{Float64}(undef, num_constraints)

    copyto!(d_variable_lower_bound, problem.variable_lower_bound)
    copyto!(d_variable_upper_bound, problem.variable_upper_bound)
    copyto!(d_isfinite_variable_lower_bound, isfinite_variable_lower_bound)
    copyto!(d_isfinite_variable_upper_bound, isfinite_variable_upper_bound)
    copyto!(d_objective_vector, problem.objective_vector)
    copyto!(d_right_hand_side, problem.right_hand_side)

    # * A
    d_constraint_matrix = CUDA.CUSPARSE.CuSparseMatrixCSR(problem.constraint_matrix)
    # * Aᵀ
    d_constraint_matrix_t = CUDA.CUSPARSE.CuSparseMatrixCSR(problem.constraint_matrix')

    return CuLinearProgrammingProblem(
        num_variables,
        num_constraints,
        d_variable_lower_bound,
        d_variable_upper_bound,
        d_isfinite_variable_lower_bound,
        d_isfinite_variable_upper_bound,
        d_objective_vector,
        problem.objective_constant,
        d_constraint_matrix,
        d_constraint_matrix_t,
        d_right_hand_side,
        problem.num_equalities,
    )
end


"""
Transfer scaled QP from CPU to GPU
"""
function scaledqp_cpu_to_gpu(scaled_problem::ScaledQpProblem)
    d_constraint_rescaling = CuArray{Float64}(
        undef,length(scaled_problem.constraint_rescaling)
    )
    d_variable_rescaling = CuArray{Float64}(
        undef,length(scaled_problem.variable_rescaling)
    )

    # 将向量 constrain/variable 从CPU复制到GPU
    copyto!(d_constraint_rescaling, scaled_problem.constraint_rescaling)
    copyto!(d_variable_rescaling, scaled_problem.variable_rescaling)

    # 将原始问题、缩放问题、记录了缩放问题的两个向量从CPU复制到GPU
    return CuScaledQpProblem(
        qp_cpu_to_gpu(scaled_problem.original_qp),
        qp_cpu_to_gpu(scaled_problem.scaled_qp),
        d_constraint_rescaling,
        d_variable_rescaling,
    )
end

"""
Transfer solutions from GPU to CPU

- `d_primal_solution → primal_solution`
- `d_dual_solution → dual_solution`
"""
function gpu_to_cpu!(
    d_primal_solution::CuVector{Float64},
    d_dual_solution::CuVector{Float64},
    primal_solution::Vector{Float64},
    dual_solution::Vector{Float64},
)
    copyto!(primal_solution, d_primal_solution)
    copyto!(dual_solution, d_dual_solution)
end


