# Mix-cuPDLP.jl

Utilizing `cuPDLP` (a GPU-friendly LP solver) from https://github.com/jinwen-yang/cuPDLP.jl, I developed a mixed linear programming solver using the branch-and-bound method. Additionally, I experimented with various initialization and pruning techniques.

## Setup

A one-time step is required to set up the necessary packages on the local machine:

```shell
$ julia --project -e 'import Pkg; Pkg.instantiate()'
```

## Running 

`solve_mip.jl` is the recommended script for using cuPDLP. The results are written to JSON and text files. All commands below assume that the current directory is the working directory.

```shell
$ julia --project scripts/solve_mip.jl \
--instance_path=INSTANCE_PATH \
--output_directory=OUTPUT_DIRECTORY \
--tolerance=TOLERANCE \
--time_sec_limit=TIME_SEC_LIMIT
```

## Branch and Bound
- Queue type for tree search
  - FIFO (BFS)
  - FILO (DFS)
  - Best first search (priority queue，use `lower_bound` to compare)
- The root node is the LP problem after relaxation of the original MIP problem
- The dynamically updated `iteration_limit` is used to limit `cuPDLP` because it is possible for `cuPDLP` to iterate *tens of times the normal number of iterations* before realizing that a problem is not feasible
- `upper_bound`: The global upper bound, which is the minimum of the solution satisfying all integer constraints
- `lower_bound`: The lower bound of each node, is `obj` of the relaxed LP problem of its parent
- `lower_bounds`: A hash map, stored as a map between the **index** of all nodes present on the B&B tree (i.e. in the queue) and the corresponding `lower_bound`
- For each loop:
  - Get a node
  - relax and use `cuPDLP` to solve the LP problem
  - Obtain output and update global `upper_bound` etc
  - If it needs to branch, modify the node to get two children
- Pruning condition
  - Infeasible
  - Feasible and satisfies all the integer constraints
  - Feasible but `lower_bound > upper_bound`
- Stopping criterion
  - Optimal: `|upper_bound - min_lower_bound| < eps`
  - End of exploration: The queue is empty