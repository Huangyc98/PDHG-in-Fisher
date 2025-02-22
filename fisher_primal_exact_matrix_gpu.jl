# GPU_id = 0
# ENV["CUDA_VISIBLE_DEVICES"] = "$GPU_id"

using LinearAlgebra
using SparseArrays
using Printf
using Random
using JuMP
using Ipopt
using Plots

import CUDA
using CUDA


include("model_definition.jl")
include("solver_gpu_dense.jl")
include("test_exact.jl")

function main()
# 主程序：调用生成和求解过程
n = 100 # 消费者数量
m = 100   # 商品数量
max_outer_iterations = 1000
Random.seed!(6)

# m = Int(m)
# n = Int(n)
# 生成随机问题
problem,agents = generate_log_problem_with_agent(m, n)
d_problem = fisher_cpu_to_gpu(problem)
println("problem_generate_success")
# # 使用Ipopt解问题
#  Ipopt_time = @elapsed begin
#  model = Model(Ipopt.Optimizer)
#  optimalvalue = solve_with_ipopt(model, problem.w, u, problem.beq)
#  end
# # 调用 PDHG 算法进行优化

println("\nRunning PDHG optimization...")
pdhg_time = @elapsed begin
        solverstate,bufferstate,Restart_info = problem_initialize(d_problem)
stop_criteria = 1e-2
residuals_plot = Float64[]
w_plot = Float64[]
frac = Float64[]
old_current_dual_solution= CUDA.zeros(m)
    for i=1:max_outer_iterations

        solverstate,bufferstate,Restart_info = PDHG_gpu_adaptive_restart(d_problem,solverstate,bufferstate,Restart_info,stop_criteria)
        D_exact = exact_max_deficiency(agents, Array(solverstate.current_dual_solution[1:m]))
        println(D_exact)
        # println(sum(solverstate.current_dual_solution[1:m]))
        # solverstate.current_dual_solution[1:m] = solverstate.current_dual_solution[1:m]*m/sum(-solverstate.current_dual_solution[1:m])
        residual_dual = d_problem.e*solverstate.current_dual_solution[1:m]+d_problem.w

        residual_w = old_current_dual_solution .- solverstate.current_dual_solution[1:m]
        # println("w: ",d_problem.w)
        # println("dual: ",-solverstate.current_dual_solution[1:m])
        residual_rel = norm(residual_dual)/maximum(d_problem.w)
        # residual_abs = norm(residual_dual)
        # w_abs = norm(residual_w)
        # push!(residuals_plot, residual_abs) 
        # push!(w_plot, w_abs)
        # push!(frac,residual_abs/w_abs)
        println("residual_dual: ",residual_rel)
        if residual_rel <=1e-4
            break
        end
        # sleep(3)
        d_problem.w .= -d_problem.e*solverstate.current_dual_solution[1:m]
        old_current_dual_solution .= solverstate.current_dual_solution[1:m]
        stop_criteria =max(0.75* stop_criteria,1e-4)
        # stop_criteria = 1e-4
        println("iteration: ",i)
    end
    # plot(1:length(residuals_plot), residuals_plot, xlabel="Iteration", ylabel="Residual", title="Residual Convergence", yscale=:log10,legend=false,label="p")
    # plot!(1:length(w_plot), w_plot, label="w")
    # savefig("p_w.png") 

    # plot(1:length(frac), frac, xlabel="Iteration", ylabel="p/w", title="Residual Convergence", yscale=:log10,legend=false)
    # savefig("frac.png") 
end
println("norm_w: ",CUDA.norm(d_problem.w))
println("PDHG optimization completed.")

# println("Optimal allocation (x_ij) of PDHG:")
# println(x_opt_PDHG)
# println("Maximized utility of solver:")
# println(primal_vals_PDHG[end])
println("PDHG runtime (seconds): ", pdhg_time)
# println("Ipopt value: ",optimalvalue)
# println("Ipopt time: ",Ipopt_time)
# plot_results(primal_vals_PDHG, dual_vals_PDHG)
end


function PDHG_gpu_adaptive_restart(d_problem::cuFisherproblem, solverstate::cuPdhgSolverState, bufferstate::cuBufferstate,Restart_info::cuRestartinfo,stop_criteria,iteration_limit=1000000)
    #问题输入
    m = d_problem.m
    n = d_problem.n
    initial_primal_dual_weight = 1.0/sqrt(m+n)
    Residual = residual(
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    # 用于存储每次迭代的目标函数值
    primal_residual_val = []
    dual_residual_val = []
    primal_dual_gap_val = []
    max_residual = []
    Residual1 = []
    Residual2 = []
    iteration = 0
    println("initialize success")
    
    while true
        iteration +=1

        if mod(iteration,120)==0||iteration == iteration_limit + 1 || solverstate.numerical_error

            Residual = compute_residual_matrix_gpu(solverstate,d_problem,bufferstate.eq2_paramter)
            current_residual = max(Residual.current_primal_residual,Residual.current_dual_residual, Residual.current_gap)
            avg_residual = max(Residual.avg_primal_residual,Residual.avg_dual_residual, Residual.avg_gap)
            # println("current_residual: ",current_residual)
            # println("avg_residual: ",avg_residual)
            primal_residual = min(current_residual,avg_residual)
            # println("solverstate.inner_iteration: ",solverstate.inner_iteration)

            better_residual = min(current_residual,avg_residual)
            if better_residual <=0.2*Restart_info.last_res_residual
                Restart_Flag = true
                # println("restart_condition 1")
            else
                if better_residual <=0.8*Restart_info.last_res_residual && better_residual>=Restart_info.last_residual
                Restart_Flag = true
                # println("restart_condition 2")
                else
                    if solverstate.inner_iteration>=0.2*solverstate.total_number_iterations
                    Restart_Flag = true
                    # println("restart_condition 3")
                    else
                        Restart_Flag = false
                        # println(" no restart")
                    end
                end
            end

            Restart_info.last_residual=better_residual
            if Restart_Flag
                # println("do restart")
                if current_residual <= avg_residual
                    solverstate.avg_dual_solution .= solverstate.current_dual_solution
                    solverstate.avg_primal_solution_x.nzVal .= solverstate.current_primal_solution_x.nzVal
                    solverstate.avg_primal_solution_t .= solverstate.current_primal_solution_t
                    # println("current")
                     Restart_info.last_res_residual = current_residual
                    # copyto!(Restart_info.last_res_residual,current_residual)

                else
                    # println("avg")
                    # sleep(1)
                    solverstate.current_dual_solution .=solverstate.avg_dual_solution
                    solverstate.current_primal_solution_x.nzVal .= solverstate.avg_primal_solution_x.nzVal
                    solverstate.current_primal_solution_t .= solverstate.avg_primal_solution_t
                    # copyto!(Restart_info.last_res_residual,avg_residual)
                     Restart_info.last_res_residual = avg_residual
                end
                solverstate.inner_iteration = 1

                Restart_info.res_primal_distence = max(CUDA.norm(Restart_info.last_res_primal_x0.nzVal - solverstate.current_primal_solution_x.nzVal),
                CUDA.norm(Restart_info.last_res_primal_t0 - solverstate.current_primal_solution_t))
                Restart_info.res_dual_distence = CUDA.norm(Restart_info.last_res_dual - solverstate.current_dual_solution)
                # println("res_primal_distence: ",Restart_info.res_primal_distence)
                # println("res_dual_distence: ",Restart_info.res_dual_distence)

                Restart_info.last_res_primal_x0.nzVal .= solverstate.current_primal_solution_x.nzVal
                Restart_info.last_res_primal_t0 .= solverstate.current_primal_solution_t
                Restart_info.last_res_dual .= solverstate.current_dual_solution
                
                
                solverstate.primal_weight  = compute_new_primal_weight(Restart_info,solverstate.primal_weight,d_problem.m)
                solverstate.primal_weight = min(max(solverstate.primal_weight,1e-4*initial_primal_dual_weight),100*initial_primal_dual_weight)
                # println("solverstate.primal_weight: ", solverstate.primal_weight)

            end
            # println(Residual.current_primal_residual)
            # println(Residual.current_dual_residual)
            # println(Residual.current_gap)
            # @printf("Iteration %d, obj_value: %e\n, primal_residual: %e, dual_residual: %e, complementary: %e\n", 
            # iteration, -sum(d_problem.w .* log.(sum(d_problem.u.*solverstate.current_primal_solution,dims=2))),
            # Residual.current_primal_residual,Residual.current_dual_residual, Residual.current_gap)
            if mod(iteration,360)==0
                @printf("Iteration %d, obj_value: %e\n, primal_residual: %e, dual_residual: %e, complementary: %e\n", 
                iteration,  -CUDA.sum(d_problem.w .* log.(solverstate.current_primal_solution_t)),Residual.current_primal_residual,Residual.current_dual_residual, Residual.current_gap)
            end           
             #println(maximum(solverstate.current_primal_solution[1:d_problem.m]))
            if primal_residual<=stop_criteria || iteration >= iteration_limit+1
                break
            end


        end

        take_step_exact_matrix_gpu!(d_problem,solverstate,bufferstate)

        weight = 1 / (1.0 + solverstate.inner_iteration)

        solverstate.avg_primal_solution_x.nzVal .+=  weight*(solverstate.current_primal_solution_x.nzVal .- solverstate.avg_primal_solution_x.nzVal)
        solverstate.avg_primal_solution_t .+=weight*(solverstate.current_primal_solution_t .-solverstate.avg_primal_solution_t)
        solverstate.avg_dual_solution .+= weight*(solverstate.current_dual_solution.-solverstate.avg_dual_solution)
        solverstate.inner_iteration += 1
        
        # Residual,residual_1,residual_2 = compute_residual_matrix_gpu(solverstate,d_problem)
        # push!(Residual1,residual_1)
        # push!(Residual2,residual_2)
        
        # residual1 = compute_residual_gpu(solverstate,d_problem)
        # push!(primal_residual_val,residual1.current_primal_residual)
        # push!(dual_residual_val,residual1.current_dual_residual)
        # push!(primal_dual_gap_val,residual1.current_gap)
        # push!(max_residual,max(residual1.current_primal_residual,residual1.current_dual_residual,residual1.current_gap))
    end
    # it_del = 1
    # println("Residual1 length: ", length(Residual1))
    # println("Residual2 length: ", length(Residual2))
    # println("Residual1[it_del:end]: ", Residual1[it_del:end])
    # println("Residual2[it_del:end]: ", Residual2[it_del:end])
    # p = plot(
    #     1: 10000, 
    #     Residual1[it_del:end], 
    #     label="Residual_eq_1", 
    #     title="Residual Trajectory", 
    #     xlabel="Iteration", 
    #     ylabel="Residual_eq_1 Value", 
    #     yscale=:log10,
    #     legend=:topright
    # )
    # plot!(
    #     1:10000,  # 确保 x 轴范围一致
    #     Residual2[it_del:end],
    #     label="Residual_eq_2",
    # )
    
    # savefig(p, "eq_residual_trajectory.png")
    return solverstate,bufferstate,Restart_info
end

main()