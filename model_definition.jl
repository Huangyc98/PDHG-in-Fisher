mutable struct Restartinfo
    last_res_primal::CuArray{Float64}
    last_res_dual::CuArray{Float64}
    res_primal_distence::Float64
    res_dual_distence::Float64
    
end

mutable struct cuRestartinfo
    last_res_primal_x0::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}
    last_res_primal_t0::CuArray{Float64}
    last_res_dual::CuArray{Float64}
    res_primal_distence::Float64
    res_dual_distence::Float64
    last_res_residual::Float64
    last_residual::Float64
end

mutable struct Fisherproblem
    w::Vector{Float64}
    u::SparseMatrixCSC{Float64,Int64}
    beq::Vector{Float64}
    e::SparseMatrixCSC{Float64,Int64}
    ori_m::Int64
    ori_n::Int64
    x0::SparseMatrixCSC{Float64,Int64}
    t0::Vector{Float64}
end

mutable struct cuFisherproblem
    w::CuArray{Float64}          # 权重
    u::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}  # 效用矩阵
    beq::CuArray{Float64}        # 等式约束
    e::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}   # 不等式约束
    m::Int64                       # 问题规模
    n::Int64
    x0::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}    # 初始猜测
    t0::CuArray{Float64}
end

mutable struct cuPdhgSolverState
    current_primal_solution_x::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}
    current_primal_solution_t::CuArray{Float64}
    current_dual_solution::CuArray{Float64}
    prev_primal_solution_x::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}
    prev_primal_solution_t::CuArray{Float64}
    prev_dual_solution::CuArray{Float64}
    avg_primal_solution_x::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}
    avg_primal_solution_t::CuArray{Float64}
    avg_dual_solution::CuArray{Float64}
    step_size::Float64
    primal_weight::Float64
    total_number_iterations::Int64
    numerical_error::Bool
    inner_iteration::Int64
end

mutable struct PdhgSolverState
    current_primal_solution::Vector{Float64}
    current_dual_solution::Vector{Float64}
    prev_primal_solution::Vector{Float64}
    prev_dual_solution::Vector{Float64}
    avg_primal_solution::Vector{Float64}
    avg_dual_solution::Vector{Float64}
    avg_dual_product::Vector{Float64}
    current_dual_product::Vector{Float64}
    step_size::Float64
    primal_weight::Float64
    total_number_iterations::Int64
    numerical_error::Bool
    inner_iteration::Int64
end

mutable struct Bufferstate
    primal_solution::Vector{Float64}
    dual_solution::Vector{Float64}
    dual_product::Vector{Float64}
    step_size::Float64
    primal_weight::Float64
end

mutable struct cuBufferstate
    primal_solution_x::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}
    primal_solution_t::CuArray{Float64}
    dual_solution::CuArray{Float64}
    step_size::Float64
    primal_weight::Float64
    ux::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}
    eq2_paramter::Float64
end

mutable struct residual
    current_primal_residual::Float64
    current_dual_residual::Float64
    current_gap::Float64
    avg_primal_residual::Float64
    avg_dual_residual::Float64
    avg_gap::Float64
end