# identification of the vector field values of a nonlinear networked system (dual method)

using DifferentialEquations
using LinearAlgebra
using Random

# dynamics
n = 30
nb_interactions = 5
power_max = 3
amplitude_coeff = 1.
A_lin = -1. *rand(n)
F = zeros(nb_interactions,5,n)
for dim = 1 : n
    F[:,:,dim] = [zeros(Int,nb_interactions,4) amplitude_coeff*randn(nb_interactions,1)]
    for j = 1 : nb_interactions
        tot_deg = rand(2:power_max)
        first_deg = rand(1:tot_deg)
        F[j,1:4,dim] = [randperm(n)[1:2]' first_deg tot_deg-first_deg]
    end
end
p = [A_lin, F, n]

function dyn(du,u,p,t)
A_lin = p[1]
F = p[2]
n = p[3]
for dim = 1 : n
    du[dim] = A_lin[dim]*u[dim]
end
for dim = 1 : n, k = 1 : size(F,1)

    du[dim] = du[dim] + F[k,5,dim]*u[convert(Int,F[k,1,dim])]^F[k,3,dim]*u[convert(Int,F[k,2,dim])]^F[k,4,dim]

end
end

F_coeff = Vector{Array{Float64, 2}}(undef, n)
for dim = 1 : n

    F_coeff[dim]=[zeros(size(F,1),n) F[:,5,dim]]

    for k = 1 : size(F,1)

        F_coeff[dim][k,convert(Int,F[k,1,dim])]=F[k,3,dim]
        F_coeff[dim][k,convert(Int,F[k,2,dim])]=F[k,4,dim]

    end

    F_coeff[dim] = [F_coeff[dim];[zeros(1,n) A_lin[dim]]]
    F_coeff[dim][end,dim] = 1

end

# parameters
deg_F = 3
t_end = 1.
nb_step = 2
nb_simus = 200
ic_range = (-0.5, 0.5)
sigma_noise = 0.01

basis = 0 # 0 for GRBF
gam = 0.01 # GRB parameter
factor = 1.
deg_F = 3

# simulation
time_step = t_end / nb_step
nb_samples = nb_simus * nb_step
tspan = (0.0, t_end)
u0 = ic_range[1] .+ rand(nb_simus,n) * (ic_range[2]-ic_range[1])

X = zeros(nb_samples, n)
Y = zeros(nb_samples, n)
for j in 1 : nb_simus

  prob = ODEProblem(dyn,u0[j,:],tspan,p)
  sol = solve(prob,Tsit5(),saveat=time_step,reltol=1e-6,abstol=1e-50)

  if nb_step == 1

      X[j,:] = sol[1:n,1]'
      Y[j, :] = sol[1:n,2]'

  else

      X[(j-1)*nb_step+1 : j*nb_step, :] = sol[1:n,1:end-1]'
      Y[(j-1)*nb_step+1 : j*nb_step, :] = sol[1:n,2:end]'

  end

end

X = X + sigma_noise.*randn(size(X)).*X
Y = Y + sigma_noise.*randn(size(Y)).*Y

# identification

F_samples = lifting_ident_dual(X, Y, time_step, factor = factor, basis = basis, gamma = gam)

# verification

F_exact = zeros(nb_samples, n)
for j = 1 : nb_samples

    for dim = 1 : n
        F_exact[j,dim] = A_lin[dim]*X[j,dim]
    end
    for dim = 1 : n, k = 1 : size(F,1)

        F_exact[j,dim] = F_exact[j,dim] + F[k,5,dim]*X[j,convert(Int,F[k,1,dim])]^F[k,3,dim]*X[j,convert(Int,F[k,2,dim])]^F[k,4,dim]

    end

end

using Statistics
RMSE = norm(F_samples[:]-F_exact[:])/sqrt(n*nb_samples)
NRMSE = RMSE/mean(abs.(F_exact[:]))

# figure
using Plots
pyplot()
plt = quiver(X[:,1], X[:,2], gradient = (F_samples[:,1] ./5, F_samples[:,2] ./5), label = "exact")
plt = quiver!(X[:,1], X[:,2], gradient = (F_exact[:,1] ./5, F_exact[:,2] ./5), color = :red, label = "estimated", legend = :best, xlabel = "x_1", ylabel = "x_2")

# uncomment to estimate the vector field coefficients using the Lasso method

# # construct library functions
# F_samples = F_samples/factor
# X = X/factor
#
# nb_dic_F = convert(Int, prod(n+1:n+deg_F)/factorial(deg_F))
#
# index = [zeros(Int,1,n);Matrix{Int}(I,n,n)]
#
# stack = Dict(dim => convert(Array{Int,2},hcat(index[dim+1,:])') for dim = 1 : n)
#
# for k = 2 : deg_F, dim = 1 : n
#
#     global current_stack = Array{Int}(undef,0,n)
#     for dim2 = dim : n
#
#         global current_stack = [current_stack;stack[dim2]+ones(Int,size(stack[dim2],1),1)*index[dim+1,:]']
#
#     end
#
#     stack[dim] = current_stack
#     global index = [index;current_stack]
#
# end
#
# stack = 0
# current_stack = 0
#
# psi_X = zeros(nb_samples,nb_dic_F)
# for k = 1 : nb_dic_F
#
#     psi_X[:,k]=prod(X.^(ones(nb_samples,1)*index[k,:]'),dims=2)
#
# end
#
# # optimization (find vector field coefficients)
#
# using Lasso
# lambda = 2e-4
#
# coeff = zeros(nb_dic_F, n)
# for d = 1 : n
#
#     path = fit(LassoPath, psi_X, F_samples[:,d], λ=[lambda], standardize = false, intercept = false) # cd_tol = 1e-4
#     coeff[:,d] = convert(Array,path.coefs)
#
#     # x = Variable(nb_dic_F)
#     # problem = minimize(sumsquares(psi_X * x - F_samples[:,dim]) + lambda*norm(x,1))
#     # solve!(problem)
#     # problem.status # :Optimal, :Infeasible, :Unbounded etc.
#     # coeff[:,dim] = x.value
#     println(d)
#
# end
#
# # correction scaling
#
# for k = 1 : nb_dic_F
#
#     deg = sum(index[k,:])
#     coeff[k,:] = coeff[k,:].*factor^(1-deg)
#
# end
#
# # verification
# coeff_exact = zeros(nb_dic_F, n)
# for dim = 1 : n, j = 1 : size(F_coeff[dim], 1)
#     ss = convert(Int,sum(F_coeff[dim][j, 1:n]))
#     if ss <= deg_F
#         if ss > 0
#             index_min = convert(Int,prod(n+1:n+ss-1)/factorial(ss-1)+1)
#             index_max = convert(Int,prod(n+1:n+ss)/factorial(ss))
#         else
#             index_min = 1
#             index_max = 1
#         end
#         val = falses(nb_dic_F)
#         val[index_min:index_max] = [index[i,:]==F_coeff[dim][j,1:n] for i = index_min:index_max]
#         coeff_exact[val,dim] .= F_coeff[dim][j,n+1]
#     end
# end
#
# index_non_zero = (LinearIndices(abs.(coeff_exact).>0))[findall(abs.(coeff_exact).>0)]
#
#
# # figure
# using Plots
# ENV["GKSwstype"]="gksqt"
# pyplot()
# K = 20
# plt = plot(index_non_zero[index_non_zero.<K*nb_dic_F], coeff_exact[index_non_zero[index_non_zero.<K*nb_dic_F]], seriestype=:scatter, markershape = :circle, markersize = 5, msw = 2, markercolor = :white, msc = :blue, label = "exact")
# plt = plot!(1:K*nb_dic_F,coeff[1:K*nb_dic_F], lw = 1, color = :red, label = "estimated", legend = :best, xlabel = "Index", ylabel = "\$ w_k^j\$", tickfontsize = 15, legendfontsize = 14, guidefont = 20, xticks = 0:1e5:K*nb_dic_F)
# display(plt)
#
# using Statistics
# RMSE = norm(coeff-coeff_exact)/sqrt(n*nb_dic_F)
# NRMSE = RMSE/mean(abs.(coeff_exact[index_non_zero]))
#
# println("RMSE: ", RMSE)
# println("NRMSE: ", NRMSE)
