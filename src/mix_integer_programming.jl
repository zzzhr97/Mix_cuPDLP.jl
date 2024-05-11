

"""
A MixIntegerProgrammingProblem struct specifies a mixed interger programming 
problem with the following format:

```
minimize 1/2 x' * objective_matrix * x + objective_vector' * x
		  + objective_constant

s.t.constraint_matrix[1:num_equalities, :] * x =
	right_hand_side[1:num_equalities]

	constraint_matrix[(num_equalities + 1):end, :] * x >=
	right_hand_side[(num_equalities + 1):end]

	variable_lower_bound <= x <= variable_upper_bound

    integer_variables[i] = 0 for continuous variables 
    and 1 for integer variables.
```
The variable_lower_bound may contain `-Inf` elements and variable_upper_bound
may contain `Inf` elements when the corresponding variable bound is not present.
"""
mutable struct MixIntegerProgrammingProblem 
    """
    The slacked linear programming problem to be solved.
    """
    slacked_problem::QuadraticProgrammingProblem
    """
    The 0-1 vector of integer variables, 0 for continuous and 1 for integer.
    """
    integer_variables::Vector{Int64}
end
    
mutable struct BranchAndBoundNodeNumber
    """
    The index of the node in the branch and bound tree.
    """
    idx::Int64
end

"""
A BranchAndBoundNode struct specifies a node in the branch and bound tree.
"""
mutable struct BranchAndBoundNode
    """
    The index of the node in the branch and bound tree.
    """
    index::Int64
    """
    The MIP problem to be solved.
    """
    problem::MixIntegerProgrammingProblem
    """
    The lower bound of this branch, which is the obj of the slacked problem
    of the parent node. If the lower bound >= upper bound, the node will 
    be pruned. (Upper bound is stored externally.)
    """
    lower_bound::Float64
end

mutable struct MIPOutput
    primal_solution::Vector{Float64}
    dual_solution::Vector{Float64}
    termination_reason::TerminationReason
    iteration_count::Int32
    primal_objective::Float64
end

function get_MIPOutput(
    output::SaddlePointOutput
)::MIPOutput
    return MIPOutput(
        output.primal_solution,
        output.dual_solution,
        output.termination_reason,
        output.iteration_count,
        output.iteration_stats[end].convergence_information[1].primal_objective
    )
end

"""
A BranchAndBoundControlBuffer struct specifies a buffer for the B&B alg.
"""
mutable struct BranchAndBoundControlBuffer
    """
    The index of the best node in the B&B tree before.
    """
    best_idx::Union{Int64, Nothing}
    """
    The upper bound of the B&B tree.
    """
    upper_bound::Float64
    """
    The existed lower bounds of the B&B tree, which is to get minimum value 
    of all the nodes' lower bound.
    """
    lower_bounds::Dict{Int64, Float64}
    """
    The key of the minimum value of the lower_bounds.
    """
    min_lower_bound_idx::Union{Int64, Nothing}
    """
    The best solution info of the B&B tree.
    """
    best_info::Union{Nothing, MIPOutput}
end

