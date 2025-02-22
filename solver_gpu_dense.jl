
function generate_log_problem(m, n)
    # 随机生成权重 w 和效用矩阵 u
    e = sprand(n,m,0.25)
    # e = e./(sum(e,dims=1))
    # println("e: ", sum(e,dims=1))
    w = Array{Float64}(undef, n)
    w .= sum(e,dims=2)  # 随机生成 m 个权重
    # println("w: ", w) 
    u = sprand(n, m,0.5)   # 随机生成 m x n 的效用矩阵
    # println("u: ", sum(u,dims=2))
    # u = rand(n, m)
    # 初始猜测
    x0 = similar(u)   # 初始化 x
    # x0 .= (u'.*sum(e,dims=1)')'
    vec = sum(e,dims=1) ./ sum(u,dims=1)
    # x0 = u
    x0 .= u.*vec
    # x0 .= (u./sum(u,dims=1)).*sum(e,dims=1) 
    # println(size(u./sum(u,dims=1)))
    # println(size(sum(e,dims=1)))
    # println("x0 ", size(x0))
    # 生成线性约束矩阵 Aeq 和向量 beq
    # A1 = kron(diagm(ones(Float64, n)), ones(1, m))
    t0 = Array{Float64}(undef, n)
    # t0 .= sum(u.*x0,dims =2)
    t0 .= sum(x0.*u,dims=2)
    b = Array{Float64}(undef, m)
    b .= sum(e,dims=1)'
    println(norm(b))
    # P = spzeros(m, n * m)
    # for i in 1:m
    #     P[i, (i-1)*n+1:(i-1)*n+n] .= -u[i, :]
    # end
    problem = Fisherproblem(
        w,u,b,e,m,n,x0,t0
    )
    return problem,u
end

function generate_log_problem_with_agent(m, n)
    # 随机生成权重 w 和效用矩阵 u
    e = sprand(n,m,0.25)
    # e = e./(sum(e,dims=1))
    # println("e: ", sum(e,dims=1))
    w = Array{Float64}(undef, n)
    w .= sum(e,dims=2)  # 随机生成 m 个权重
    # println("w: ", w) 
    u = sprand(n, m,0.5)   # 随机生成 m x n 的效用矩阵
    # println("u: ", sum(u,dims=2))
    # u = rand(n, m)
    # 初始猜测
    x0 = similar(u)   # 初始化 x
    # x0 .= (u'.*sum(e,dims=1)')'
    vec = sum(e,dims=1) ./ sum(u,dims=1)
    # x0 = u
    x0 .= u.*vec
    # x0 .= (u./sum(u,dims=1)).*sum(e,dims=1) 
    # println(size(u./sum(u,dims=1)))
    # println(size(sum(e,dims=1)))
    # println("x0 ", size(x0))
    # 生成线性约束矩阵 Aeq 和向量 beq
    # A1 = kron(diagm(ones(Float64, n)), ones(1, m))
    t0 = Array{Float64}(undef, n)
    # t0 .= sum(u.*x0,dims =2)
    t0 .= sum(x0.*u,dims=2)
    b = Array{Float64}(undef, m)
    b .= sum(e,dims=1)'
    println(norm(b))
    # P = spzeros(m, n * m)
    # for i in 1:m
    #     P[i, (i-1)*n+1:(i-1)*n+n] .= -u[i, :]
    # end
    agents = Vector{Agent}(undef, n)
    for i in 1:n
        agents[i] = Agent((e[i, :]),(u[i, :]))
    end
    problem = Fisherproblem(
        w,u,b,e,m,n,x0,t0
    )
    return problem,agents
end

function fisher_cpu_to_gpu(problem::Fisherproblem)
    
    d_w = CuArray{Float64}(undef, length(problem.w))
    d_beq = CuArray{Float64}(undef, length(problem.beq))
    d_t0 = CuArray{Float64}(undef, length(problem.w))

    copyto!(d_w, problem.w)
    copyto!(d_beq,problem.beq)
    copyto!(d_t0,problem.t0)

    d_x0 = CUDA.CUSPARSE.CuSparseMatrixCSR(problem.x0)
    d_u =  CUDA.CUSPARSE.CuSparseMatrixCSR(problem.u)
    d_e = CUDA.CUSPARSE.CuSparseMatrixCSR(problem.e)
    return cuFisherproblem(d_w,d_u,d_beq,d_e,problem.ori_m,problem.ori_n,d_x0,d_t0)
end

    function take_step_exact_matrix_gpu!(
        problem::cuFisherproblem,
        solverstate::cuPdhgSolverState,
        bufferstate::cuBufferstate,        
        )

        step_size = solverstate.step_size
        m = problem.m
        n = problem.n

        done = false
        k = 1
        while !done
            k += 1
            if k>=20
                break
            end
            tau = solverstate.step_size /solverstate.primal_weight
            sigma = solverstate.step_size * solverstate.primal_weight

            # println("primal1: ",solverstate.current_primal_solution)
            # println("dual1: ",solverstate.current_dual_solution)
            # sleep(2)
            # println("x0_new: ",typeof(CUDA.sum(solverstate.current_primal_solution_x,dims=1)))
            # CUDA.sum(2.0.*solverstate.current_primal_solution_x-solverstate.current_primal_solution_x,dims=2)
            # println("t_1: ",Array(bufferstate.primal_solution_t))
            # println("x_1: ",Array(solverstate.current_primal_solution_x))
            # sleep(3)

            bufferstate.primal_solution_x, bufferstate.primal_solution_t = primal_exact_solution_matrix_gpu(
                problem.w, 
                problem.u, 
                solverstate.current_primal_solution_x, 
                solverstate.current_primal_solution_t,
                solverstate.current_dual_solution,
                tau, 
                problem.m, 
                problem.n,
                bufferstate.eq2_paramter,
                bufferstate.primal_solution_x, 
                bufferstate.primal_solution_t)
            # println("t_2: ",Array(bufferstate.primal_solution_t))
            # println("x_2: ",Array(bufferstate.primal_solution_x))
            # sleep(3)
            
            #dual update
            #  println("delta primal: ",CUDA.norm(bufferstate.primal_solution-solverstate.current_primal_solution))
            
            # grad_1 = CUDA.sum(solverstate.current_primal_solution_x,dims=1) 
            # grad_2 = CUDA.zeros(Float64, problem.m)
            # println("y_1: ",Array(solverstate.current_dual_solution))
            grad_1 = CUDA.sum(2.0.*bufferstate.primal_solution_x-solverstate.current_primal_solution_x,dims=1)'-problem.beq
            # println(size(2.0 *bufferstate.primal_solution_x.nzVal .- solverstate.current_primal_solution_x.nzVal))
            # println(size(problem.u.nzVal))
            bufferstate.ux.nzVal = problem.u.nzVal .* (2.0 *bufferstate.primal_solution_x.nzVal .- solverstate.current_primal_solution_x.nzVal)
            grad_2 = 2.0.*bufferstate.primal_solution_t .- solverstate.current_primal_solution_t .- CUDA.sum(bufferstate.ux,dims=2)
            bufferstate.dual_solution .= solverstate.current_dual_solution - sigma .*[grad_1;bufferstate.eq2_paramter.*grad_2]
            # println("y_2: ",Array(bufferstate.dual_solution))
            # bufferstate.dual_solution .= solverstate.current_dual_solution - sigma .*(problem.Aeq*(bufferstate.primal_solution) -problem.beq)
            
            # println("primal2: ",solverstate.current_primal_solution)
            # println("dual2: ",solverstate.current_dual_solution)
            # sleep(2)

            interaction, movement = compute_interaction_and_movement_gpu(
            solverstate,
            bufferstate,
            problem,
            )
            #  println("interaction: ",interaction)
            #  println("movement: ",movement)
            if interaction > 0
                step_size_limit = movement / interaction
                if movement == 0.0
                    solverstate.numerical_error = true
                    break
                end
            else
                step_size_limit = Inf
            end
        
                if step_size <= step_size_limit
                update_solution_in_solver_state_gpu!(
                    solverstate, 
                    bufferstate,
                )
                done = true
                end
                # println("step_size_limit: ",step_size_limit)
                # println("step_size: ",step_size)
            first_term = (1 - 1/(solverstate.inner_iteration + 1)^(0.3)) * step_size_limit
            second_term = (1 + 1/(solverstate.inner_iteration + 1)^(0.6)) * step_size
            step_size = min(first_term, second_term)
            step_size = min(max(step_size,0.1/sqrt(m+n)),4.0/sqrt(m+n))
                # println(step_size)
        end
               solverstate.step_size = step_size
    end

function primal_exact_solution_matrix_gpu(
        w::CuArray{Float64},
        u::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32},
        x0_init::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32},
        t0_init::CuArray{Float64},
        y::CuArray{Float64},
        tau::Float64,
        m::Int64,
        n::Int64,
        delta::Float64,
        x0_new::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32},
        t0_new::CuArray{Float64},
    )
        # lambda = CuArray{Float64}(undef, length(m))
        # y2 = CuArray{Float64}(undef, length(n))
        lambda = @view y[1:m]
        y2 = @view y[m+1:m+n]
        y2 = delta.*y2
        # lambda = @view y[1:m]
        # y2 = @view y[m+1:m+n]
        # println("x0_new: ",typeof(CUDA.sum(x0_init,dims=2)))
        size_row = length(u.rowPtr)-1        
        threads_per_block = 256
        num_blocks = cld(size_row, threads_per_block)
        # println(typeof(lambda))
        @cuda threads=threads_per_block blocks=num_blocks log_primal_update_in_cuda(u.rowPtr, u.colVal, size_row, u.nzVal, tau, w,lambda,y2,  t0_init,x0_init.nzVal, t0_new,x0_new.nzVal)
        # println("x0_new: ",typeof(CUDA.sum(x0_new,dims=2)))
        return x0_new,t0_new
    end

function compute_interaction_and_movement_gpu(
    solverstate::cuPdhgSolverState,
    bufferstate::cuBufferstate,
    problem::cuFisherproblem,
)
    delta_primal_x = similar(solverstate.current_primal_solution_x)
    delta_primal_x.nzVal = bufferstate.primal_solution_x.nzVal .- solverstate.current_primal_solution_x.nzVal
    bufferstate.ux.nzVal = problem.u.nzVal.*(delta_primal_x.nzVal)

    delta_primal_t = similar(solverstate.current_primal_solution_t)
    delta_primal_t = bufferstate.primal_solution_t .- solverstate.current_primal_solution_t
    delta_primal2 = delta_primal_t - CUDA.sum(bufferstate.ux,dims=2)

    delta_dual = bufferstate.dual_solution .- solverstate.current_dual_solution
    # println("delta_primal: ",CUDA.norm(delta_primal))
    primal_dual_interaction = CUDA.dot([CUDA.sum(delta_primal_x,dims=1)';bufferstate.eq2_paramter.*delta_primal2], delta_dual)
    interaction = abs(primal_dual_interaction)
    norm_delta_primal = CUDA.norm([CUDA.sum(delta_primal_x,dims=1)';bufferstate.eq2_paramter.*delta_primal2],2)
    norm_delta_dual = CUDA.norm(delta_dual)
        
    movement = 0.5*(solverstate.primal_weight * norm_delta_primal^2 + (1 / solverstate.primal_weight) * norm_delta_dual^2)
    return interaction, movement
end

function update_solution_in_solver_state_gpu!(
    solverstate::cuPdhgSolverState,
    bufferstate::cuBufferstate,
)

    solverstate.current_primal_solution_x.nzVal .= bufferstate.primal_solution_x.nzVal
    solverstate.current_primal_solution_t .= bufferstate.primal_solution_t
    solverstate.current_dual_solution .= bufferstate.dual_solution

end

function compute_new_primal_weight(
    last_restart_info::cuRestartinfo,
    primal_weight::Float64,
    m::Int64,
    primal_weight_update_smoothing::Float64=0.1,
)
    primal_distance = last_restart_info.res_primal_distence
    dual_distance = last_restart_info.res_dual_distence
    # println("primal_distance: ",primal_distance)
    # println("dual_distance: ", dual_distance)
    if primal_distance > eps() && dual_distance > eps()
        new_primal_weight_estimate = dual_distance / (primal_distance)
        # Exponential moving average.
        # If primal_weight_update_smoothing = 1.0 then there is no smoothing.
        # If primal_weight_update_smoothing = 0.0 then the primal_weight is frozen.
        log_primal_weight =
            primal_weight_update_smoothing * log(new_primal_weight_estimate) +
            (1 - primal_weight_update_smoothing) * log(primal_weight)

        primal_weight = exp(log_primal_weight)

        return primal_weight
    else
        return primal_weight
    end
end

function compute_residual_matrix_gpu(
    solverstate::cuPdhgSolverState,
    problem::cuFisherproblem,
    eq2_paramter,
    )
    m = problem.m
    n = problem.n
    ux = similar(problem.u)
    ux.nzVal .=  solverstate.current_primal_solution_x.nzVal.* problem.u.nzVal
    y2 = @view solverstate.current_dual_solution[m+1:m+n]
    y2 = eq2_paramter.*y2
    current_primal_residual_lhs = [CUDA.sum(solverstate.current_primal_solution_x,dims=1)';(solverstate.current_primal_solution_t-CUDA.sum(ux,dims=2))]
    current_primal_residual = CUDA.norm([CUDA.sum(solverstate.current_primal_solution_x,dims=1)' .- problem.beq;(solverstate.current_primal_solution_t-CUDA.sum(ux,dims=2))],Inf)/(1+max(CUDA.norm(current_primal_residual_lhs,Inf),CUDA.norm(problem.beq,Inf)))
    # residual_1 = CUDA.norm(CUDA.sum(solverstate.current_primal_solution_x,dims=1)' .- problem.beq,Inf)/(1+max(CUDA.norm(current_primal_residual_lhs,Inf),CUDA.norm(problem.beq,Inf)))
    # residual_2 = CUDA.norm((solverstate.current_primal_solution_t-CUDA.sum(ux,dims=2)),Inf)/(1+max(CUDA.norm(current_primal_residual_lhs,Inf),CUDA.norm(problem.beq,Inf)))
    # println("residual_1: ",CUDA.norm(CUDA.sum(solverstate.current_primal_solution_x,dims=1)' .- problem.beq,Inf)/(1+max(CUDA.norm(current_primal_residual_lhs,Inf),CUDA.norm(problem.beq,Inf))))
    # println("residual_2: ",CUDA.norm((solverstate.current_primal_solution_t-CUDA.sum(ux,dims=2)),Inf)/(1+max(CUDA.norm(current_primal_residual_lhs,Inf),CUDA.norm(problem.beq,Inf))))

    lambda_yu, wt = compute_log_residual(solverstate.current_primal_solution_t,problem.u,solverstate.current_dual_solution,problem.w,eq2_paramter)
    current_dual_residual_abs = max(maximum(-lambda_yu.nzVal),maximum(wt+y2),0.0)
    current_dual_norm_1 = CUDA.norm(wt,Inf)
    current_dual_norm_2 = max(maximum(CUDA.abs.(lambda_yu.nzVal)),maximum(CUDA.abs.(y2)))
    current_dual_residual_relative = current_dual_residual_abs/(1+max(current_dual_norm_1,current_dual_norm_2))
    # println("dual_residual_relative: ",dual_residual_relative)
    
    # gap_matrix = similar(problem.u)
    current_gap_matrix_nzVal = max.(lambda_yu.nzVal,0.0).*solverstate.current_primal_solution_x.nzVal
    current_gap_vec = max.(-wt-y2,0.0).*solverstate.current_primal_solution_t
    current_gap_abs = max(maximum(current_gap_matrix_nzVal),maximum(current_gap_vec),0.0)
    current_gap_norm_1 = max(maximum(max.(lambda_yu.nzVal,0.0)),maximum(max.(-wt-y2,0.0)))
    current_gap_norm_2 = max(maximum(solverstate.current_primal_solution_x.nzVal),maximum(solverstate.current_primal_solution_t))
    current_gap_relative = current_gap_abs/(1+current_gap_norm_1+current_gap_norm_2)
    
    ux.nzVal .=  solverstate.avg_primal_solution_x.nzVal.* problem.u.nzVal
    y2 = @view solverstate.avg_dual_solution[m+1:m+n]
    y2 = eq2_paramter.*y2
    avg_primal_residual_lhs = [CUDA.sum(solverstate.avg_primal_solution_x,dims=1)';(solverstate.avg_primal_solution_t-CUDA.sum(ux,dims=2))]
    avg_primal_residual = CUDA.norm([CUDA.sum(solverstate.avg_primal_solution_x,dims=1)' .- problem.beq;(solverstate.avg_primal_solution_t-CUDA.sum(ux,dims=2))],Inf)/(1+max(CUDA.norm(avg_primal_residual_lhs,Inf),CUDA.norm(problem.beq,Inf)))
    
    lambda_yu, wt = compute_log_residual(solverstate.avg_primal_solution_t,problem.u,solverstate.avg_dual_solution,problem.w,eq2_paramter)
    avg_dual_residual_abs = max(maximum(-lambda_yu.nzVal),maximum(wt+y2),0.0)
    avg_dual_norm_1 = CUDA.norm(wt,Inf)
    avg_dual_norm_2 = max(maximum(CUDA.abs.(lambda_yu.nzVal)),maximum(CUDA.abs.(y2)))
    avg_dual_residual_relative = avg_dual_residual_abs/(1+max(avg_dual_norm_1,avg_dual_norm_2))
    # println("dual_residual_relative: ",dual_residual_relative)
    
    # gap_matrix = similar(problem.u)
    avg_gap_matrix_nzVal = max.(lambda_yu.nzVal,0.0).*solverstate.avg_primal_solution_x.nzVal
    avg_gap_vec = max.(-wt-y2,0.0).*solverstate.avg_primal_solution_t
    avg_gap_abs = max(maximum(avg_gap_matrix_nzVal),maximum(avg_gap_vec),0.0)
    avg_gap_norm_1 = max(maximum(max.(lambda_yu.nzVal,0.0)),maximum(max.(-wt-y2,0.0)))
    avg_gap_norm_2 = max(maximum(solverstate.avg_primal_solution_x.nzVal),maximum(solverstate.avg_primal_solution_t))
    avg_gap_relative = avg_gap_abs/(1+avg_gap_norm_1+avg_gap_norm_2)


    return residual(current_primal_residual,current_dual_residual_relative,current_gap_relative,avg_primal_residual,avg_dual_residual_relative,avg_gap_relative)
end

function compute_log_residual(
    t::CuVector{Float64},
    u::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32},
    y::CuVector{Float64},
    w::CuVector{Float64},
    eq2_paramter::Float64,
    )
    n,m = size(u)
    rowptr = u.rowPtr  # 行指针
    colind = u.colVal  # 列索引
    # nzval = u.nzVal   # 非零值
    size_row = length(rowptr) - 1        
    threads_per_block = 256
    num_blocks = cld(size_row, threads_per_block)

    lambda = @view y[1:m]
    y2 = @view y[m+1:m+n]
    y2 = eq2_paramter.*y2
    wt = similar(w)
    lambda_yu = similar(u)
    
    @cuda threads=threads_per_block blocks=num_blocks dual_residual_kernel(rowptr, colind,size_row, u.nzVal, w, t,lambda,y2,wt,lambda_yu.nzVal)
    return lambda_yu, wt
end

function log_primal_update_in_cuda(rowptr,colind,size_row,u,tau,w,lambda,y2,t0_init,x0_init,t0_new,x0_new)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size_row
        # 从 GPU 上读取 rowptr 的值
        
        t0_new[i] = (t0_init[i] + tau * y2[i] + sqrt((tau* y2[i]+t0_init[i])^2+4*tau*w[i]))/2

        start_idx = rowptr[i]
        end_idx = rowptr[i + 1] - 1

        # 遍历该行的非零列
        for j = start_idx:end_idx
            #x0_new[j] = max(tau*( lambda[colind[j]] - y2[i]*u[j]),0)
            x0_new[j] = max(x0_init[j] + tau*( lambda[colind[j]] - y2[i]*u[j] ), 0 )
        end
    end
    return
end


function dual_residual_kernel(rowptr,colind,size_row,u,w,t,lambda,y,wt,lambda_yu)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size_row
        # 从 GPU 上读取 rowptr 的值
        
        wt[i] = w[i]/t[i]

        start_idx = rowptr[i]
        end_idx = rowptr[i + 1] - 1

        # 遍历该行的非零列
        for j = start_idx:end_idx
            #x0_new[j] = max(tau*( lambda[colind[j]] - y2[i]*u[j]),0)
           lambda_yu[j] = -lambda[colind[j]] + y[i]*u[j]
        end
    end
    return 
end

function problem_initialize(d_problem::cuFisherproblem)
    m = d_problem.m
    n = d_problem.n
    x0 = d_problem.x0
    t0 = d_problem.t0
    dual_size = n+m
    initial_stepsize = 0.1/sqrt(m+n)
    println("initial_stepsize: ",initial_stepsize)
    initial_primal_dual_weight = 1.0/sqrt(m+n)

    solverstate = cuPdhgSolverState(
        1.0.*x0,
        1.0.*t0,
        CUDA.ones(Float64, dual_size),
        0.5.*x0,
        0.5.*t0,
        CUDA.ones(Float64, dual_size),
        1.0.*x0,
        1.0.*t0,
        CUDA.ones(Float64, dual_size),
        initial_stepsize,
        initial_primal_dual_weight,
        1,
        false,
        1,
    )

    bufferstate = cuBufferstate(
        1.0.*x0,
        1.0.*t0,
        CUDA.ones(Float64, dual_size),
        initial_stepsize,
        initial_primal_dual_weight,
        1.0.*x0,
        sqrt(n/m),
    )
    # sqrt(n/m),
    Restart_info = cuRestartinfo(
        1.0.*x0,
        1.0.*t0,
        CUDA.ones(Float64,dual_size),
        0.0,
        0.0,
        10.0,
        10.0,
    )
    return solverstate,bufferstate,Restart_info
end