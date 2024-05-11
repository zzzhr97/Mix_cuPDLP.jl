

function print_bb_info(
    parameters_mip::MIPParameters,
    n_iteration::Int64,
    n_nodes::BranchAndBoundNodeNumber,
    control_buffer::BranchAndBoundControlBuffer,
    cur_node::BranchAndBoundNode,
    cur_output::MIPOutput,
    cur_status::BranchAndBoundNodeStatus,
    iteration_limit::Int64 = typemax(Int32),
    start_time::Float64 = time()
)   
    if parameters_mip.verbosity >= 3
        println("")
        println("Iteration: \t<", n_iteration, ">")
        println("Total node number: ", n_nodes.idx)

        println("Control buffer: ")
        println("\tbest_idx: ", control_buffer.best_idx)
        println("\tupper_bound: ", control_buffer.upper_bound)
        println("\tmin_lower_bound_idx: ", control_buffer.min_lower_bound_idx)
        println("\tmin_lower_bound: ", 
            control_buffer.min_lower_bound_idx,
            "->",
            control_buffer.lower_bounds[control_buffer.min_lower_bound_idx]
        )

        println("Current node: ")
        println("\tindex: ", cur_node.index)
        println("\tlower_bound: ", cur_node.lower_bound)

        println("Current output: ")
        println("\ttermination_reason: ", cur_output.termination_reason)
        println("\titeration_count: ", cur_output.iteration_count)
        println("\tobjective: ", 
            cur_output.primal_objective
        )
        if length(cur_output.primal_solution) <= 10
            println("\tprimal_solution: ", cur_output.primal_solution)
        end

        println("Current satisfy vector: ")
        println("\t", check_integer_condition(
            cur_output.primal_solution,
            cur_node.problem.integer_variables
        ))

        println("Current status: ", cur_status)

        println("Iteration limit: ", iteration_limit)

        println("Time: ", time() - start_time)
        println()
    
    elseif parameters_mip.verbosity >= 2

        satisfy_vector = check_integer_condition(
            cur_output.primal_solution,
            cur_node.problem.integer_variables
        )

        @assert isempty(control_buffer.lower_bounds) == false
        Printf.@printf(
            "Iter[%s]  Node[%s]  UB[%.6f]  LB[%.6f]  CurLB[%.6f]  \nSatisfied[%s]  IterLimit[%s]  Time[%.2e]  CurStatus[%s]\n",
            lpad(n_iteration, 6, '0'),
            lpad(n_nodes.idx, 6, '0'),
            control_buffer.upper_bound,
            control_buffer.lower_bounds[control_buffer.min_lower_bound_idx],
            cur_node.lower_bound,
            lpad(sum(satisfy_vector), 4, '0'),
            lpad(iteration_limit, 7, '0'),
            (time() - start_time),
            cur_status,
        )
        println()

    end
end

function is_integer(
    x::Float64,
    atol::Float64 = 1e-6
)::Bool
    return isapprox(x, round(x), atol=atol)
end

"""
Check each value of the primal solution satisfies integer condition or not.
"""
function check_integer_condition(
    primal_solution::Vector{Float64},
    integer_variables::Vector{Int64},
    eps_round::Float64 = 1e-6
)::Vector{Bool}
    #!此处的位运算优先级高于逻辑运算 !!!
    return (integer_variables .== 0.0) .| is_integer.(primal_solution, eps_round)
end

"""
Check a primal solution satisfies integer condition or not.
"""
function check_integer_condition_all(
    primal_solution::Vector{Float64},
    integer_variables::Vector{Int64},
    eps_round::Float64 = 1e-6
)::Bool
    return (
        all(
            (integer_variables .== 0.0) .|
            is_integer.(primal_solution, eps_round)
        ) 
    )
end

"""
Round the satisfied integer variables of the primal solution.
Like 5.999999999999997 -> 6.0
"""
function round_integer_all!(
    primal_solution::Vector{Float64},
    integer_variables::Vector{Int64},
    satisfy_vector::Vector{Bool},
    eps_round::Float64 = 1e-6
)
    len = length(primal_solution)
    @inbounds for i in 1:len
        if integer_variables[i] == 1 && satisfy_vector[i]
            primal_solution[i] = round(
                primal_solution[i], 
                digits=Int64(round(-log10(eps_round)))
            )
        end
    end
end

"""
Update control buffer, return the status of the current node.
- Update best_info / upper_bound / best_idx if the new solution is better.
- Update lower_bound of the current node.
"""
function update_control_buffer!(
    control_buffer::BranchAndBoundControlBuffer,
    cur_node::BranchAndBoundNode,
    cur_output::MIPOutput,
    atol_check::Float64 = 1e-5   # 最优gap的检查精度以及 lower_bound 和 obj 的检查精度
)::BranchAndBoundNodeStatus

    # feasible
    if cur_output.termination_reason == TERMINATION_REASON_OPTIMAL

        cur_obj = cur_output.primal_objective
        
        # 检查所有整数条件是否满足
        is_satisfy = check_integer_condition_all(
            cur_output.primal_solution,
            cur_node.problem.integer_variables
        )

        # 更新当前结点的 lower_bound
        # 父结点的 lower_bound 一定比当前结点的 obj 小
        # @assert isapprox(max(cur_node.lower_bound-cur_obj, 0.0), 0.0, atol=atol_check)
        cur_node.lower_bound = cur_obj

        # 满足整数条件
        if is_satisfy
            # 还没有全局最优解，或者找到全局更好的解，更新 upper_bound
            if (
                isinf(control_buffer.upper_bound) || 
                cur_obj < control_buffer.upper_bound
            )
                control_buffer.best_info = cur_output
                control_buffer.best_idx = cur_node.index
                control_buffer.upper_bound = cur_obj

                # 检查 upper_bound 与当前最小的 lower_bound 之间的差异
                # ub < lb + abs(lb) * atol_check
                # 如果满足，找到全局最优解
                # 或者lower_bounds为空，也找到全局最优解
                min_lower_bound = control_buffer.lower_bounds[
                    control_buffer.min_lower_bound_idx]
                if (
                    isinf(min_lower_bound) ||
                    control_buffer.upper_bound <= (
                        min_lower_bound +
                        abs(min_lower_bound) * atol_check
                    )
                )
                    return BB_NODE_OPTIMAL
                end
            end
            
            # 满足整数条件，剪枝
            return BB_NODE_PRUNED_SATISFIED

        # 不满足整数条件
        else
            # lower bound 比全局 upper_bound 大，剪枝
            if cur_obj >= control_buffer.upper_bound
                return BB_NODE_PRUNED_BOUND

            # 正常情况，不剪枝
            else
                return BB_NODE_NO_PRUNED
            end
        end

    # infeasible
    else
        return BB_NODE_PRUNED_INFEASIBLE
    end
end

"""
Given one deleted node, update the lower bounds 
hash map and the minimum lower bound index.
"""
function del_lower_bounds!(
    control_buffer::BranchAndBoundControlBuffer,
    del_node::BranchAndBoundNode
)
    del_node_idx = del_node.index
    delete!(control_buffer.lower_bounds, del_node_idx)
    if isempty(control_buffer.lower_bounds) == true
        control_buffer.min_lower_bound_idx = nothing
    elseif del_node_idx == control_buffer.min_lower_bound_idx
        control_buffer.min_lower_bound_idx = (
            argmin(control_buffer.lower_bounds)
        )
    end
end

"""
Given two added nodes, update the lower bounds 
hash map and the minimum lower bound index.
Note: The lower bounds of these two nodes are the same.
"""
function add_lower_bounds!(
    control_buffer::BranchAndBoundControlBuffer,
    add_nodes::Vector{BranchAndBoundNode}
)
    for node in add_nodes
        control_buffer.lower_bounds[node.index] = node.lower_bound
    end

    # Julia index starts from 1
    if(isnothing(control_buffer.min_lower_bound_idx))
        control_buffer.min_lower_bound_idx = add_nodes[1].index
    else
        cur_min_lower_bound = control_buffer.lower_bounds[
            control_buffer.min_lower_bound_idx
        ]
        if(add_nodes[1].lower_bound < cur_min_lower_bound)
            control_buffer.min_lower_bound_idx = add_nodes[1].index
        end
    end
end

"""
Branching function for the B&B algorithm.
- Given a node and its slacked result.
- Return a list of new nodes. (2 nodes)
"""
function branching(
    parameters_mip::MIPParameters,
    father_node::BranchAndBoundNode,
    father_output::MIPOutput,
    n_nodes::BranchAndBoundNodeNumber,
    eps_round::Float64 = 1e-6   # 四舍五入精度
)::Vector{BranchAndBoundNode}

    # 反映是否满足整数条件的Bool向量
    satisfy_vector = check_integer_condition(
        father_output.primal_solution,
        father_node.problem.integer_variables
    )

    # 将满足整数条件的整数变量按精度四舍五入
    # 如 5.999999999999997 四舍五入到 6.0
    round_integer_all!(
        father_output.primal_solution,
        father_node.problem.integer_variables,
        satisfy_vector
    )

    # 找到第一个不满足整数条件的变量索引
    cur_idx = findfirst(.!satisfy_vector)

    @assert isnothing(cur_idx) == false
    
    # 找到这个变量的值，并进行四舍五入
    cur_value = father_output.primal_solution[cur_idx]
    cur_value = round(cur_value, digits=Int64(round(-log10(eps_round))))

    # 向下和向上取整
    floor_value = floor(cur_value)
    ceil_value = ceil(cur_value)
    cur_variable_lower_bound = (
        father_node.problem.slacked_problem.variable_lower_bound[cur_idx]
    )
    cur_variable_upper_bound = (
        father_node.problem.slacked_problem.variable_upper_bound[cur_idx]
    )

    if parameters_mip.verbosity >= 3
        println("\n", "-"^40)
        println("cur_idx: ", cur_idx)
        println("cur_value: ", cur_value)
        println(father_node.index, "->", n_nodes.idx+1, " ", n_nodes.idx+2)
        println(cur_variable_lower_bound)
        println(cur_variable_upper_bound)
        println(floor_value)
        println(ceil_value)
        println("-"^40, "\n")
    end

    @assert (
        (
            isinteger(cur_variable_lower_bound) && (
                isinteger(cur_variable_upper_bound) ||
                isinf(cur_variable_upper_bound)
            ) 
        ) &&
        (
            cur_variable_lower_bound <= 
            floor_value <= 
            ceil_value <= 
            cur_variable_upper_bound
        ) &&
        (
            cur_variable_lower_bound !=
            cur_variable_upper_bound
        )
    )

    # 深拷贝 branching
    new_nodes = BranchAndBoundNode[
        BranchAndBoundNode(
            n_nodes.idx + 1,
            deepcopy(father_node.problem),
            father_node.lower_bound
        ),
        BranchAndBoundNode(
            n_nodes.idx + 2,
            deepcopy(father_node.problem),
            father_node.lower_bound
        )
    ]
    n_nodes.idx += 2

    # 修改子结点的上下界
    new_nodes[1].problem.slacked_problem.variable_upper_bound[cur_idx] = floor_value
    new_nodes[2].problem.slacked_problem.variable_lower_bound[cur_idx] = ceil_value

    return new_nodes
end

function push_buffer!(
    tree_buffer::Union{
        PriorityQueue{BranchAndBoundNode, Float64}, 
        Queue{BranchAndBoundNode},
        Stack{BranchAndBoundNode}
    },
    node::BranchAndBoundNode
)
    if isa(tree_buffer, PriorityQueue)
        enqueue!(tree_buffer, node, node.lower_bound)
    elseif isa(tree_buffer, Queue)
        enqueue!(tree_buffer, node)
    else
        push!(tree_buffer, node)
    end
end

function pop_buffer!(
    tree_buffer::Union{
        PriorityQueue{BranchAndBoundNode, Float64}, 
        Queue{BranchAndBoundNode},
        Stack{BranchAndBoundNode}
    }
)::BranchAndBoundNode
    if isa(tree_buffer, Stack)
        return pop!(tree_buffer)
    else
        return dequeue!(tree_buffer)
    end
end

"""
从 buffer 中获取结点，直到拿到 n_pop_nodes 个结点
如果返回空数组，说明 buffer 中的所有结点的 lower_bound 
均比全局 upper_bound 大，则终止 B&B 算法
"""
function get_nodes_from_buffer(
    tree_buffer::Union{
        PriorityQueue{BranchAndBoundNode, Float64}, 
        Queue{BranchAndBoundNode},
        Stack{BranchAndBoundNode}
    },
    control_buffer::BranchAndBoundControlBuffer,
    n_pop_nodes::Int64
)::Vector{BranchAndBoundNode}
    nodes = BranchAndBoundNode[]

    while n_pop_nodes > 0
        if isempty(tree_buffer)
            break
        end

        cur_node = pop_buffer!(tree_buffer)

        # 检查新拿出来的结点的 lower bound (由父节点计算得出) 
        # 是否比全局 upper_bound 还大
        # 之所以会出现这种情况，是因为在这个结点加入到队列中之后
        # upper_bound 更新变小了
        if cur_node.lower_bound >= control_buffer.upper_bound
            continue
        else
            push!(nodes, cur_node)
            n_pop_nodes -= 1
        end
    end

    return nodes
end

function solve_current_nodes(
    parameters_lp::PdhgParameters,
    parameters_mip::MIPParameters,
    tree_buffer::Union{
        PriorityQueue{BranchAndBoundNode, Float64}, 
        Queue{BranchAndBoundNode},
        Stack{BranchAndBoundNode}
    },
    control_buffer::BranchAndBoundControlBuffer,
    n_pop_nodes::Int64
)

    # 从 buffer 中获取结点
    cur_nodes::Vector{BranchAndBoundNode} = get_nodes_from_buffer(
        tree_buffer,
        control_buffer,
        n_pop_nodes
    )

    cur_nodes_num::Int64 = length(cur_nodes)
    println(">"^10, "\t\tCur nodes num: ", cur_nodes_num)

    if cur_nodes_num == 0
        # buffer 为空，返回空数组，终止 B&B 算法
        return cur_nodes, MIPOutput[]

    elseif cur_nodes_num == 1

        # 只有一个结点，直接求解
        return cur_nodes, MIPOutput[
            get_MIPOutput(
                optimize(
                    parameters_lp, 
                    cur_nodes[1].problem.slacked_problem,
                    parameters_mip
                )
            )
        ]
    else 

        # 多个结点，合并成一个问题，再求解
        return cur_nodes, multi_optimize(
            parameters_lp, 
            parameters_mip,
            cur_nodes
        )
    end
end

"""
动态更新 iteration limit
会改变 parameters_lp.termination_criteria.iteration_limit
"""
function update_limit_iteration!(
    parameters_lp::PdhgParameters,
    parameters_mip::MIPParameters,
    iteration_limit::Int64,
    termination_reason::TerminationReason,
    n_iteration::Int64,
    iteration_count::Int32
)
    if parameters_mip.limit_iteration
        λ::Float64 = 0.0
        if iteration_limit < 1
            λ = 0.0
        elseif termination_reason == TERMINATION_REASON_OPTIMAL
            if n_iteration < 20
                λ = 0.8
            else
                λ = 0.95
            end
        else
            λ = 1.0
        end
        iteration_limit = ceil(
            iteration_limit * λ + 
            iteration_count * (1 - λ)
        )

        # 实际的 iteration limit
        if n_iteration < 7
            parameters_lp.termination_criteria.iteration_limit = ceil(
                iteration_limit * (16 - n_iteration) * 0.5
            )
        else
            parameters_lp.termination_criteria.iteration_limit = ceil(
                iteration_limit * 5
            )
        end
    end

    return iteration_limit
end

"""
Branch and Bound algorithm for solving mixed integer programming problems.
"""
function bb_optimize(
    parameters_lp::PdhgParameters, 
    parameters_mip::MIPParameters,
    mip::MixIntegerProgrammingProblem
)

    n_nodes::BranchAndBoundNodeNumber = BranchAndBoundNodeNumber(1)
    n_iteration::Int64 = 0
    cur_status::BranchAndBoundNodeStatus = BB_NODE_NO_PRUNED
    cur_node = nothing
    iteration_limit::Int64 = 0
    start_time = time()

    # initialize B&B root node
    root_node = BranchAndBoundNode(
        n_nodes.idx,
        deepcopy(mip),  # 深拷贝，避免修改原始数据
        -Inf,           # 根结点的父结点是不存在的，所以 lower_bound 为 -Inf
    )

    # initialize control buffer 
    control_buffer = BranchAndBoundControlBuffer(
        nothing,
        Inf,
        Dict{Int64, Float64}([(1, -Inf)]),
        n_nodes.idx,
        nothing     # 注意是 nothing 而不是 Nothing !!! 前者是实例，后者是类型 !!!
    )

    # initialize tree buffer
    if parameters_mip.tree_buffer_type == BB_QUEUE_BEST_FIRST
        tree_buffer = PriorityQueue{BranchAndBoundNode, Float64}()
    elseif parameters_mip.tree_buffer_type == BB_QUEUE_BREADTH_FIRST
        tree_buffer = Queue{BranchAndBoundNode}()
    else
        tree_buffer = Stack{BranchAndBoundNode}()
    end
    push_buffer!(tree_buffer, root_node)

    if parameters_mip.verbosity >= 1
        println("\n", "-"^50)
        println("Begin B&B algorithm...")
    end

    # search the B&B tree
    while !isempty(tree_buffer)

        n_iteration += 1

        # 同时求解多个结点
        cur_nodes, cur_outputs = solve_current_nodes(
            parameters_lp,
            parameters_mip,
            tree_buffer, 
            control_buffer, 
            parameters_mip.combine_problem_num
        )

        # buffer 中所有剩余的结点的 lower_bound 均大于全局 upper_bound
        # 则终止 B&B 算法
        if length(cur_outputs) == 0
            break
        end

        for i in 1:length(cur_outputs)
            cur_output = cur_outputs[i]
            cur_node = cur_nodes[i]
            iteration_limit = update_limit_iteration!(
                parameters_lp,
                parameters_mip,
                iteration_limit,
                cur_output.termination_reason,
                n_iteration,
                cur_output.iteration_count,
            )

            # 更新控制buffer的 best_info / upper_bound / best_idx (如果找到更好的解)
            # 更新当前结点的 lower_bound
            # 返回当前结点的状态
            cur_status = update_control_buffer!(
                control_buffer, 
                cur_node, 
                cur_output
            )

            print_bb_info(
                parameters_mip,
                n_iteration,
                n_nodes,
                control_buffer,
                cur_node,
                cur_output,
                cur_status,
                iteration_limit,
                start_time
            )

            del_lower_bounds!(
                control_buffer, 
                cur_node
            )
            
            # 找到 global optimal solution
            if cur_status == BB_NODE_OPTIMAL
                break

            # 剪枝
            elseif cur_status in [
                BB_NODE_PRUNED_BOUND,
                BB_NODE_PRUNED_INFEASIBLE, 
                BB_NODE_PRUNED_SATISFIED
            ]
                continue

            # 不剪枝
            else
                add_nodes = branching(
                    parameters_mip,
                    cur_node,
                    cur_output,
                    n_nodes
                )

                # 在 lower_bound 哈希映射中加入新的结点
                add_lower_bounds!(
                    control_buffer,
                    add_nodes
                )

                # 将新的结点加入到队列中
                for node in add_nodes
                    push_buffer!(tree_buffer, node)
                end
            end
        end
    end

    if isnothing(control_buffer.best_idx)
        return [
            MIPOutput(
                Float64[],
                Float64[],
                TERMINATION_REASON_UNSPECIFIED,
                0,
                0.0,
            ),
            -1
        ]
    end

    # 将最优解的整数变量四舍五入
    # 如 5.999999999999997 四舍五入到 6.0
    round_integer_all!(
        control_buffer.best_info.primal_solution,
        cur_node.problem.integer_variables,
        check_integer_condition(
            control_buffer.best_info.primal_solution,
            cur_node.problem.integer_variables
        )
    )

    total_time = 0.0
    if parameters_mip.verbosity >= 1
        println("\n", "-"^50)
        println("B&B algorithm End.")
        println("Current status: ", cur_status)
        println("Total iteration: ", n_iteration)
        println("Total node number: ", n_nodes.idx)
        total_time = time() - start_time
        Printf.@printf("Total time: %.2e\n", total_time)
        println("Best objective: ", control_buffer.upper_bound)
        println("-"^50)
    end

    # return control_buffer.best_info
    return [control_buffer.best_info, total_time]
end
    