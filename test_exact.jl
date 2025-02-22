using Printf
using Random
using DataStructures  # 用于双端队列

const EPS = 1e-9

mutable struct Agent
    w::Vector{Float64}
    u::Vector{Float64}
    Agent(w, u) = new(copy(w), copy(u))
end

dot_product(a::Vector{Float64}, b::Vector{Float64}) = sum(a .* b)

function build_network_for_maxdef(agents::Vector{Agent}, p::Vector{Float64})
    m = length(p)
    n = length(agents)
    s = 0
    t = m + n + 1
    
    capacity_res = Dict{Int, Dict{Int, Float64}}()
    
    function set_cap(u::Int, v::Int, cap::Float64)
        if !haskey(capacity_res, u)
            capacity_res[u] = Dict{Int, Float64}()
        end
        capacity_res[u][v] = get(capacity_res[u], v, 0.0) + cap
        
        if !haskey(capacity_res, v)
            capacity_res[v] = Dict{Int, Float64}()
        end
        capacity_res[v][u] = get(capacity_res[v], u, 0.0)
    end
    
    # s->a_j
    for j in 1:m
        set_cap(s, j, p[j])
    end
    
    # 计算代理人的 bang-per-buck 最大值
    alpha = Vector{Float64}(undef, n)
    for i in 1:n
        best_ratio = 0.0
        for j in 1:m
            if p[j] > EPS
                ratio = agents[i].u[j] / p[j]
                if ratio > best_ratio
                    best_ratio = ratio
                end
            end
        end
        alpha[i] = best_ratio
    end
    
    INF_CAP = sum(p) * 10.0 + 10.0

    # a_j->b_i
    for i in 1:n
        for j in 1:m
            if p[j] > EPS
                ratio = agents[i].u[j] / p[j]
                if abs(ratio - alpha[i]) < 1e-12
                    aj = j
                    bi = m + i
                    set_cap(aj, bi, INF_CAP)
                end
            end
        end
    end
    
    # b_i->t
    for i in 1:n
        bi = m + i
        budget_i = dot_product(p, agents[i].w)
        set_cap(bi, t, budget_i)
    end
    
    return capacity_res, s, t
end

function build_level_graph_dinic(s::Int, t::Int, cap_res::Dict{Int, Dict{Int, Float64}}, V::Int)
    layer = fill(-1, V+5)
    layer[s+1] = 0  # Julia索引从1开始，调整顶点编号
    queue = Queue{Int}()
    enqueue!(queue, s)
    
    while !isempty(queue)
        u = dequeue!(queue)
        haskey(cap_res, u) || continue
        for v in keys(cap_res[u])
            if cap_res[u][v] > EPS && layer[v+1] == -1
                layer[v+1] = layer[u+1] + 1
                enqueue!(queue, v)
            end
        end
    end
    return layer
end

function send_flow_dinic(u::Int, flow_in::Float64, t::Int, layer::Vector{Int}, 
                        cap_res::Dict{Int, Dict{Int, Float64}}, current::Dict{Int, Int})
    u == t && return flow_in
    edges = collect(keys(get(cap_res, u, Dict{Int, Float64}())))
    while get(current, u, 1) <= length(edges)
        v = edges[get(current, u, 1)]
        if cap_res[u][v] > EPS && layer[v+1] == layer[u+1] + 1
            bottleneck = min(flow_in, cap_res[u][v])
            aug = send_flow_dinic(v, bottleneck, t, layer, cap_res, current)
            if aug > EPS
                cap_res[u][v] -= aug
                cap_res[v][u] = get(cap_res[v], u, 0.0) + aug
                return aug
            end
        end
        current[u] = get(current, u, 1) + 1
    end
    return 0.0
end

function dinic_max_flow(cap_res::Dict{Int, Dict{Int, Float64}}, s::Int, t::Int, V::Int)
    flow_val = 0.0
    while true
        layer = build_level_graph_dinic(s, t, cap_res, V)
        layer[t+1] == -1 && break
        current = Dict{Int, Int}()
        for u in keys(cap_res)
            current[u] = 1
        end
        while true
            aug = send_flow_dinic(s, Inf, t, layer, cap_res, current)
            aug < EPS && break
            flow_val += aug
        end
    end
    return flow_val
end

# ========== 实现 exact_max_deficiency 与 approximate_max_deficiency ==========

function exact_max_deficiency(agents::Vector{Agent}, p::Vector{Float64})
    m = length(p)
    n = length(agents)
    cap_res, s, t = build_network_for_maxdef(agents, p)
    
    cap_res_copy = deepcopy(cap_res)
    V = m + n + 2
    flow_val = dinic_max_flow(cap_res_copy, s, t, V)
    return flow_val
end

function build_residual_with_cutoff(cap_orig::Dict{Int, Dict{Int, Float64}}, M::Float64)
    cap_res = Dict{Int, Dict{Int, Float64}}()
    for u in keys(cap_orig)
        cap_res[u] = Dict{Int, Float64}()
        for v in keys(cap_orig[u])
            c = cap_orig[u][v]
            c_cut = c < M ? c : M
            cap_res[u][v] = c_cut
        end
    end
    # 确保反向边存在
    for u in keys(cap_res)
        for v in keys(cap_res[u])
            if !haskey(cap_res, v)
                cap_res[v] = Dict{Int, Float64}()
            end
            if !haskey(cap_res[v], u)
                cap_res[v][u] = 0.0
            end
        end
    end
    return cap_res
end

function can_push_flow_at_least(s::Int, t::Int, cap_orig::Dict{Int, Dict{Int, Float64}}, M::Float64, V::Int)
    cap_res = build_residual_with_cutoff(cap_orig, M)
    flow_val = 0.0
    while true
        layer = build_level_graph_dinic(s, t, cap_res, V)
        layer[t+1] == -1 && break
        current = Dict{Int, Int}()
        for u in keys(cap_res)
            current[u] = 1
        end
        while true
            aug = send_flow_dinic(s, Inf, t, layer, cap_res, current)
            aug < EPS && break
            flow_val += aug
            flow_val >= M && return true
        end
    end
    return flow_val >= M
end

function approximate_max_deficiency(agents::Vector{Agent}, p::Vector{Float64}; eps=1e-3)
    m = length(p)
    n = length(agents)
    cap_orig, s, t = build_network_for_maxdef(agents, p)
    V = m + n + 2
    
    L = 0.0
    R = sum(p)
    while (R - L) > eps
        M = 0.5 * (L + R)
        feasible = can_push_flow_at_least(s, t, cap_orig, M, V)
        feasible ? (L = M) : (R = M)
    end
    return 0.5 * (L + R)
end

# ========== fisher二分市场, Arrow-Debreu主算法 ==========

function construct_dichotomous_market(agents::Vector{Agent}, p::Vector{Float64}, D::Float64)
    m = length(p)
    n = length(agents)
    buyer_budgets = Vector{Float64}()
    buyer_utils = Vector{Vector{Float64}}()
    
    for i in 1:n
        e_i = dot_product(p, agents[i].w)
        push!(buyer_budgets, e_i)
        push!(buyer_utils, copy(agents[i].u))
    end
    push!(buyer_budgets, D)
    push!(buyer_utils, copy(p))
    return buyer_budgets, buyer_utils
end

function update_price_dichotomous(p_old::Vector{Float64}, buyer_budgets::Vector{Float64}, 
                                 buyer_utils::Vector{Vector{Float64}})
    p_new = copy(p_old)
    m = length(p_new)
    n = length(buyer_budgets)
    max_iter = 20
    
    for _ in 1:max_iter
        demand = zeros(m)
        for i in 1:n
            budget_i = buyer_budgets[i]
            best_ratio = 0.0
            best_indices = Int[]
            for j in 1:m
                if p_new[j] > EPS
                    ratio = buyer_utils[i][j] / p_new[j]
                    if abs(ratio - best_ratio) < 1e-12
                        push!(best_indices, j)
                    elseif ratio > best_ratio
                        best_ratio = ratio
                        best_indices = [j]
                    end
                end
            end
            isempty(best_indices) && continue
            portion = budget_i / length(best_indices)
            for j in best_indices
                demand[j] += portion / p_new[j]
            end
        end
        
        alpha = 0.05
        for j in 1:m
            if demand[j] > 1.0 + 1e-9
                p_new[j] *= (1.0 + alpha * (demand[j] - 1.0))
            end
        end
    end
    return p_new
end

function arrow_debreu_approx_equilibrium(agents::Vector{Agent}; epsilon=0.01, max_outer_iter=50)
    m = length(agents[1].w)
    n = length(agents)
    p = ones(m)
    
    for outer in 1:max_outer_iter
        D_approx = approximate_max_deficiency(agents, p, eps=epsilon*0.1)
        buyer_budgets, buyer_utils = construct_dichotomous_market(agents, p, D_approx)
        p_new = update_price_dichotomous(p, buyer_budgets, buyer_utils)
        
        # 检查预算增长
        okay = true
        for i in 1:n
            e_old = dot_product(p, agents[i].w)
            e_new = dot_product(p_new, agents[i].w)
            if e_old > EPS
                ratio = e_new / e_old
                if ratio > (1.0 + epsilon)
                    okay = false
                    break
                end
            end
        end
        
        p = copy(p_new)
        if okay
            if p[1] > EPS
                scale = 1.0 / p[1]
                p .*= scale
            end
            return p, outer
        end
    end
    
    # 超过max_outer_iter
    if p[1] > EPS
        scale = 1.0 / p[1]
        p .*= scale
    end
    return p, max_outer_iter
end

# ========== 对比 exact vs approximate ==========

function compare_exact_and_approx(agents::Vector{Agent}; p_init=nothing, eps=1e-3)
    m = length(agents[1].w)
    p_init = isnothing(p_init) ? ones(m) : copy(p_init)
    
    # 先做 exact
    t0 = time()
    D_exact = exact_max_deficiency(agents, p_init)
    exact_time = time() - t0
    
    # 再做 approximate
    t1 = time()
    D_approx = approximate_max_deficiency(agents, p_init, eps=eps)
    approx_time = time() - t1
    
    return D_exact, exact_time, D_approx, approx_time
end

function test_compare_example_small()
    agent1 = Agent([1.0, 0.0], [2.0, 1.0])
    agent2 = Agent([0.0, 1.0], [1.0, 2.0])
    agents = [agent1, agent2]
    p_init = [1.0, 1.0]
    
    D_exact, tE, D_approx, tA = compare_exact_and_approx(agents, p_init=p_init, eps=1e-5)
    println("=== test_compare_example_small ===")
    @printf(" exact D=%.4f, time=%.6f s\n", D_exact, tE)
    @printf(" approx D=%.4f, time=%.6f s\n", D_approx, tA)
    println()
end

function test_compare_example_100()
    m = 100
    n = 100
    agents = Vector{Agent}()
    for i in 1:n
        w_vec = zeros(m)
        w_vec[i] = 1.0
        u_vec = rand(m)
        push!(agents, Agent(w_vec, u_vec))
    end
    p_init = ones(m)
    
    D_exact, tE, D_approx, tA = compare_exact_and_approx(agents, p_init=p_init, eps=1e-5)
    println("=== test_compare_example_100 ===")
    @printf(" exact D=%.4f, time=%.6f s\n", D_exact, tE)
    @printf(" approx D=%.4f, time=%.6f s\n", D_approx, tA)
end

# 其他测试函数类似，根据需要补充

function main()
    test_compare_example_small()
    # test_compare_example_100()

    # D_exact = exact_max_deficiency(agents, p_init)
    # println(D_exact)
    # 其他测试函数
end

# main()