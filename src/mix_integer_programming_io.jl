#########
# Extra #
#########

@enum BranchAndBoundQueueType begin
    # Depth first search.
    BB_QUEUE_DEPTH_FIRST
    # Breath first search.
    BB_QUEUE_BREADTH_FIRST
    # Best first search.
    BB_QUEUE_BEST_FIRST
end

"""
A struct to represent a mixed integer programming problem in standard form.
"""
mutable struct MIPParameters
    """
    Whether returns GPU info or not.
    """
    mip_return::Bool
    """
    Control print level.
    - 0: no print
    - 1: print ultimate result
    - 2: print every iteration
    - 3: print details of every iteration
    """
    verbosity::Int64
    """
    Whether limit the iteration or not in each LP solving process.
    """
    limit_iteration::Bool
    """
    The type of the branch and bound queue.
    """
    tree_buffer_type::BranchAndBoundQueueType
    """
    Combine several nodes into one problem.
    """
    combine_problem_num::Int64
end

"""
No mip return, no print.
"""
function MIPParameters()
    return MIPParameters(
        false,
        0,
        false,
        BB_QUEUE_BEST_FIRST,
        1,
    )
end

"""
Reads an MPS file using the QPSReader package and transforms it into a
`MixIntegerProgrammingProblem` struct.

# Arguments
- `filename::String`: the path of the file. The file extension is ignored,
  except that if the filename ends with ".gz", then it will be uncompressed
  using GZip. Accepted formats are documented in the README of the QPSReader
  package.
- `fixed_format::Bool`: If true, parse as a fixed-format file.

# Returns
A MixIntegerProgrammingProblem struct.
"""
function qps_reader_to_standard_form_mip(
  filename::String,
  read_integer::Bool = true,
  fixed_format::Bool = false,
)::MixIntegerProgrammingProblem
    if endswith(filename, ".gz")
        io = GZip.gzopen(filename)
    else
        io = open(filename)
    end

    format = fixed_format ? :fixed : :free

    mps = Logging.with_logger(Logging.NullLogger()) do
        QPSReader.readqps(io, mpsformat = format)
    end
    close(io)

    constraint_matrix =
        sparse(mps.arows, mps.acols, mps.avals, mps.ncon, mps.nvar)

    # The reader returns only the lower triangle of the objective matrix. We have
    # to symmetrize it.
    
    obj_row_index = Int[]
    obj_col_index = Int[]
    obj_value = Float64[]
    for (i, j, v) in zip(mps.qrows, mps.qcols, mps.qvals)
        push!(obj_row_index, i)
        push!(obj_col_index, j)
        push!(obj_value, v)
        if i != j
        push!(obj_row_index, j)
        push!(obj_col_index, i)
        push!(obj_value, v)
        end
    end
    objective_matrix =
        sparse(obj_row_index, obj_col_index, obj_value, mps.nvar, mps.nvar)
    @assert mps.objsense == :notset

    slacked_problem = transform_to_standard_form(
        TwoSidedQpProblem(
            mps.lvar,
            mps.uvar,
            mps.lcon,
            mps.ucon,
            constraint_matrix,
            mps.c0,
            mps.c,
            objective_matrix
        ),
    )

    if read_integer
        integer_variables = ifelse.(mps.vartypes .== QPSReader.VTYPE_Integer, 1, 0)
    else
        integer_variables = zeros(Int64, mps.nvar)
    end

    return MixIntegerProgrammingProblem(
        slacked_problem, 
        integer_variables
    )
end