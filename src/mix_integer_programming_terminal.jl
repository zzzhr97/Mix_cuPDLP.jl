
"""
Branch and Bound algorithm control status.
This will be used to judge a node's status.
"""
@enum BranchAndBoundNodeStatus begin
    # Feasible. No pruned.
    BB_NODE_NO_PRUNED
    # Feasible. This node satisfies MIP problem.
    BB_NODE_PRUNED_SATISFIED
    # Feasible. Current branch's lower bound is greater than upper bound.
    BB_NODE_PRUNED_BOUND
    # Infeasible.
    BB_NODE_PRUNED_INFEASIBLE
    # Global optimal solution found.
    BB_NODE_OPTIMAL
end