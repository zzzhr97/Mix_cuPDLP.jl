
function combine_problem!(
    problem::QuadraticProgrammingProblem, 
    subproblem::QuadraticProgrammingProblem
)
    problem.variable_lower_bound = vcat(
        problem.variable_lower_bound,
        subproblem.variable_lower_bound
    )
    problem.variable_upper_bound = vcat(
        problem.variable_upper_bound,
        subproblem.variable_upper_bound
    )

    num_variables_first = length(problem.objective_vector)
    num_variables_second = length(subproblem.objective_vector)
    num_constraints_first = length(problem.right_hand_side)
    num_constraints_second = length(subproblem.right_hand_side)

    problem.objective_matrix = vcat(
        hcat(
            problem.objective_matrix,
            spzeros(num_variables_first, num_variables_second)
        ),
        hcat(
            spzeros(num_variables_second, num_variables_first),
            subproblem.objective_matrix
        )
    )

    problem.objective_vector = vcat(
        problem.objective_vector,
        subproblem.objective_vector
    )
    problem.objective_constant += subproblem.objective_constant

    constraint_matrix_eq = vcat(
        hcat(
            problem.constraint_matrix[1:problem.num_equalities, :],
            spzeros(problem.num_equalities, num_variables_second)
        ),
        hcat(
            spzeros(subproblem.num_equalities, num_variables_first),
            subproblem.constraint_matrix[1:subproblem.num_equalities, :]
        )
    )
    constraint_matrix_gr = vcat(
        hcat(
            problem.constraint_matrix[(problem.num_equalities + 1):end, :],
            spzeros(num_constraints_first - problem.num_equalities, num_variables_second)
        ),
        hcat(
            spzeros(num_constraints_second - subproblem.num_equalities, num_variables_first),
            subproblem.constraint_matrix[(subproblem.num_equalities + 1):end, :]
        )
    )
    
    problem.constraint_matrix = vcat(
        constraint_matrix_eq,
        constraint_matrix_gr
    )

    right_hand_side_eq = vcat(
        problem.right_hand_side[1:problem.num_equalities],
        subproblem.right_hand_side[1:subproblem.num_equalities]
    )
    right_hand_side_gr = vcat(
        problem.right_hand_side[(problem.num_equalities + 1):end],
        subproblem.right_hand_side[(subproblem.num_equalities + 1):end]
    )
    problem.right_hand_side = vcat(
        right_hand_side_eq,
        right_hand_side_gr
    )

    problem.num_equalities += subproblem.num_equalities
end

function combine_multi_problem(
    parameters_mip::MIPParameters,
    cur_nodes::Vector{BranchAndBoundNode}
)::QuadraticProgrammingProblem
    combined_problem = deepcopy(cur_nodes[1].problem.slacked_problem)
    for i in 2:length(cur_nodes)
        combine_problem!(
            combined_problem,
            cur_nodes[i].problem.slacked_problem
        )
    end
    return combined_problem
end

"""
将合并后的输出拆分成子输出，这就需要更新:
- `primal_solution`
- `dual_solution`
- `iteration_stats[end].convergence_information[1].primal_objective`
"""
function uncombine_multi_output(
    cur_nodes::Vector{BranchAndBoundNode},
    combined_output::SaddlePointOutput
)::Vector{MIPOutput}

    # ::IterationStats
    # ::ConvergenceInformation
    # combined_output.primal_solution
    # combined_output.dual_solution
    # combined_output.iteration_stats[end].convergence_information[1].primal_objective

    num_variables::Int64 = length(cur_nodes[1].problem.slacked_problem.variable_lower_bound)
    num_constraints::Int64 = length(cur_nodes[1].problem.slacked_problem.right_hand_side)
    objective_vector::Vector{Float64} = cur_nodes[1].problem.slacked_problem.objective_vector
    objective_constant::Float64 = cur_nodes[1].problem.slacked_problem.objective_constant

    total_output = MIPOutput[]
    for i in 1:length(cur_nodes)

        cur_output = MIPOutput(
            combined_output.primal_solution[1 + num_variables*(i-1) : num_variables*i],
            combined_output.dual_solution[1 + num_constraints*(i-1) : num_constraints*i],
            combined_output.termination_reason,
            combined_output.iteration_count,
            0.0
        )
        # obj = c⊺x + d
        cur_output.primal_objective = objective_vector' * cur_output.primal_solution + objective_constant
        push!(total_output, cur_output)
    end

    # for output in total_output
    #     println(output.primal_solution)
    #     println(output.dual_solution)
    #     println(output.primal_objective)
    # end

    return total_output
end

function multi_optimize(
    parameters_lp::PdhgParameters,
    parameters_mip::MIPParameters,
    cur_nodes::Vector{BranchAndBoundNode}
)::Vector{MIPOutput}
    combined_problem::QuadraticProgrammingProblem = combine_multi_problem(parameters_mip, cur_nodes)
    combined_output::SaddlePointOutput = optimize(
        parameters_lp,
        combined_problem,
        parameters_mip
    )
    
    if combined_output.termination_reason == TERMINATION_REASON_OPTIMAL

        println("Optimal solution found, uncombine the output.")
        return uncombine_multi_output(cur_nodes, combined_output)
    else

        println("Not found optimal solution, solve each subproblem.")
        total_output = MIPOutput[]
        for node in cur_nodes
            cur_output = optimize(
                parameters_lp,
                node.problem.slacked_problem,
                parameters_mip
            )
            push!(total_output, get_MIPOutput(cur_output))
        end
        return total_output
    end
end


function print_qp(
    qp::QuadraticProgrammingProblem
)
    println("variable_lower_bound: ", qp.variable_lower_bound)
    println("variable_upper_bound: ", qp.variable_upper_bound)
    println("objective_vector: ", qp.objective_vector)
    println("objective_constant: ", qp.objective_constant)
    println("constraint_matrix: ", qp.constraint_matrix)
    println("right_hand_side: ", qp.right_hand_side)
    println("num_equalities: ", qp.num_equalities)
end