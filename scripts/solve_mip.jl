import ArgParse
import GZip
import JSON3

import cuPDLP

verbosity = 2
mip_verbosity = 2
mip_combine_problem_num = 1
mip_combine_problem_num_vector = [1,2,4]

problem_num = 3
solve_num = 1

function write_vector_to_file(filename, vector)
    open(filename, "w") do io
      for x in vector
        println(io, x)
      end
    end
end

function solve_instance_and_output(
    parameters::cuPDLP.PdhgParameters,
    parameters_mip::cuPDLP.MIPParameters,
    output_dir::String,
    instance_path::String,
)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
  
    # 将路径中的文件名提取出来，把后缀去掉(如.mps)
    instance_name = replace(basename(instance_path), r"\.(mps|MPS|qps|QPS)(\.gz)?$" => "")
  
    function inner_solve()

        # 去除后缀，并转换为小写
        lower_file_name = lowercase(basename(instance_path))
        if endswith(lower_file_name, ".mps") ||
            endswith(lower_file_name, ".mps.gz") ||
            endswith(lower_file_name, ".qps") ||
            endswith(lower_file_name, ".qps.gz")
            mip = cuPDLP.qps_reader_to_standard_form_mip(instance_path)
        else
            error(
                "Instance has unrecognized file extension: ", 
                basename(instance_path),
            )
        end
    
        if parameters.verbosity >= 1
            println("Instance: ", instance_name)
        end

        # ========================================
        total_time = 0.0
        infeasible = 0
        output = nothing
        for i in 1:solve_num
            println()
            println("<"^50)
            println("Solving the ", i, "th time.")
            println(">"^50)
            println()
            output = cuPDLP.bb_optimize(parameters, parameters_mip, mip)
            if(output[2] > 0)
                total_time += output[2]
            else 
                infeasible += 1
            end
        end
        return total_time, infeasible
        # ========================================

        # * 求解问题
        # output::cuPDLP.SaddlePointOutput = cuPDLP.bb_optimize(parameters, parameters_mip, mip)
    
        if(length(output.iteration_stats) == 0)
            println("No iteration stats, something went wrong.")
            return
        end

        # 将结果写入文件 *summary.json
        log = cuPDLP.SolveLog()
        log.instance_name = instance_name
        log.command_line_invocation = join([PROGRAM_FILE; ARGS...], " ")
        log.termination_reason = output.termination_reason
        log.termination_string = output.termination_string
        log.iteration_count = output.iteration_count
        log.solve_time_sec = output.iteration_stats[end].cumulative_time_sec
        log.solution_stats = output.iteration_stats[end]
        log.solution_type = cuPDLP.POINT_TYPE_AVERAGE_ITERATE
        
        summary_output_path = joinpath(output_dir, instance_name * "_summary.json")
        open(summary_output_path, "w") do io
            write(io, JSON3.write(log, allow_inf = true))
        end
    
        # 将完整结果写入文件 *full_log.json.gz
        log.iteration_stats = output.iteration_stats
        full_log_output_path =
            joinpath(output_dir, instance_name * "_full_log.json.gz")
        GZip.open(full_log_output_path, "w") do io
            write(io, JSON3.write(log, allow_inf = true))
        end
        
        # 将原始最优解写入文件 *primal.txt
        primal_output_path = joinpath(output_dir, instance_name * "_primal.txt")
        write_vector_to_file(primal_output_path, output.primal_solution)
    
        # 将对偶解写入文件 *dual.txt
        dual_output_path = joinpath(output_dir, instance_name * "_dual.txt")
        write_vector_to_file(dual_output_path, output.dual_solution)
    end     

    return inner_solve()
end

# Warm up the GPU
function warm_up(
    lp::cuPDLP.QuadraticProgrammingProblem,
    verbosity::Int64,
)
    restart_params = cuPDLP.construct_restart_parameters(
        cuPDLP.ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        cuPDLP.KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        0.36,                   # artificial_restart_threshold
        0.2,                    # sufficient_reduction_for_restart
        0.8,                    # necessary_reduction_for_restart
        0.5,                    # primal_weight_update_smoothing
    )

    termination_params_warmup = cuPDLP.construct_termination_criteria(
        # optimality_norm = L2,
        eps_optimal_absolute = 1.0e-4,
        eps_optimal_relative = 1.0e-4,
        eps_primal_infeasible = 1.0e-12,
        eps_dual_infeasible = 1.0e-12,
        time_sec_limit = Inf,
        iteration_limit = 10,
        kkt_matrix_pass_limit = Inf,
    )

    params_warmup = cuPDLP.PdhgParameters(
        10,
        false,
        1.0,
        1.0,
        true,
        verbosity,     
        true,
        40,
        termination_params_warmup,
        restart_params,
        cuPDLP.AdaptiveStepsizeParams(0.3,0.6),
    )

    if verbosity >= 1
        println("\nWarming up GPU...")
    end

    cuPDLP.optimize(params_warmup, lp);
end


function parse_command_line()
    arg_parse = ArgParse.ArgParseSettings()

    ArgParse.@add_arg_table! arg_parse begin
        "--instance_path"
        help = "The path to the instance to solve in .mps.gz or .mps format."
        arg_type = String
        required = true

        "--output_directory"
        help = "The directory for output files."
        arg_type = String
        required = true

        "--tolerance"
        help = "KKT tolerance of the solution."
        arg_type = Float64
        default = 1e-4

        "--time_sec_limit"
        help = "Time limit."
        arg_type = Float64
        default = 3600.0
    end

    return ArgParse.parse_args(arg_parse)
end


function main()
    parsed_args = parse_command_line()
    instance_path = parsed_args["instance_path"]
    tolerance = parsed_args["tolerance"]
    time_sec_limit = parsed_args["time_sec_limit"]
    output_directory = parsed_args["output_directory"]

    lp = cuPDLP.qps_reader_to_standard_form(instance_path)

    # 求解一次固定的LP问题，用于warm up GPU
    warm_up(
        lp,
        0,      # <debug> verbosity 最初为0
    );

    restart_params = cuPDLP.construct_restart_parameters(
        cuPDLP.ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        cuPDLP.KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        0.36,                   # artificial_restart_threshold
        0.2,                    # sufficient_reduction_for_restart
        0.8,                    # necessary_reduction_for_restart
        0.5,                    # primal_weight_update_smoothing
    )

    termination_params = cuPDLP.construct_termination_criteria(
        # optimality_norm = L2,     # 默认为 L2
        eps_optimal_absolute = tolerance,
        eps_optimal_relative = tolerance,
        eps_primal_infeasible = 1.0e-8,
        eps_dual_infeasible = 1.0e-8,
        time_sec_limit = time_sec_limit,
        iteration_limit = typemax(Int32),
        kkt_matrix_pass_limit = Inf,
    )

    params = cuPDLP.PdhgParameters(
        10,
        false,
        1.0,
        1.0,
        true,
        verbosity,      # 为2，和warm up时的参数不同，那里为0
        true,
        40,
        termination_params,
        restart_params,
        cuPDLP.AdaptiveStepsizeParams(0.3,0.6),  
    )

    params_mip = cuPDLP.MIPParameters(
        true,
        mip_verbosity,
        true,
        # cuPDLP.BB_QUEUE_BEST_FIRST
        # cuPDLP.BB_QUEUE_BREADTH_FIRST
        # cuPDLP.BB_QUEUE_DEPTH_FIRST,
        cuPDLP.BB_QUEUE_BEST_FIRST,
        mip_combine_problem_num,
    )


    # ==========================================
    avg_time_vector = Float64[]
    infeasible_vector = Int64[]
    for num in mip_combine_problem_num_vector
        params_mip.combine_problem_num = num
        avg_time = 0.0
        infeasible = 0
        for i in 1:problem_num
            println(">"^20, "\tSolving instance: test_", i-1, ".mps")
            instance_path = joinpath("test/example/test_" * string(i-1) * ".mps")
            params.termination_criteria.iteration_limit = typemax(Int32)
            output = solve_instance_and_output(
                params,             # PdhgParameters
                params_mip,         # MIPParameters
                output_directory,
                instance_path,
            )
            avg_time += output[1]
            infeasible += output[2]
        end
        push!(avg_time_vector, avg_time / (problem_num * solve_num * 1.0 - infeasible))
        push!(infeasible_vector, infeasible)
    end
    for i in eachindex(mip_combine_problem_num_vector)
        println(">"^50)
        cuPDLP.Printf.@printf(
            "Combine num: %d\nAverage time: %.2e\nInfeasible: %d\n",
            mip_combine_problem_num_vector[i],
            avg_time_vector[i],
            infeasible_vector[i],
        )
    end
    println("<"^50)
    # ==========================================

    # output = solve_instance_and_output(
    #     params,             # PdhgParameters
    #     params_mip,         # MIPParameters
    #     output_directory,
    #     instance_path,
    # )

    # <debug>
    cuPDLP.debug_print("Restart Info: ")

end

main()