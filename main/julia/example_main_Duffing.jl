# Identification of the forced Duffing system (main method)
# \dot{x} = y; \dot{y} = x-x^3-0.2 y + 0.2 x^2 cos(t)

using DifferentialEquations
using LinearAlgebra
using SpecialFunctions

# dynamics
n = 2
n_tot = 3
p = [0.2, 0.2, 1.]
# p = [2., 0.3]
function dyn(du,u,p,t)
du[1] = u[2] # theta
du[2] = u[1] - u[1]^3 - p[1]*u[2] + p[2]*u[1]^2*cos(p[3]*t)
end
F_coeff = [ [0 1 0 1.],
[1 0 0 1.;3 0 0 -1.;0 1 0 -p[1];2 0 1 p[2]] ]


# parameters
deg_F = 3
t_end = 10.
nb_step = 50
nb_simus = 5
ic_range = (-1, 1)
sigma_noise = 0.01

# simulation
time_step = t_end / nb_step
nb_samples = nb_simus * nb_step
tspan = (0.0, t_end)
u0 = ic_range[1] .+ rand(nb_simus,n) .* (ic_range[2]-ic_range[1])

X = zeros(nb_samples, n_tot)
Y = zeros(nb_samples, n_tot)
for j in 1 : nb_simus

  prob = ODEProblem(dyn,u0[j,:],tspan,p)
  sol = solve(prob,Tsit5(),saveat=time_step,reltol=1e-9,abstol=1e-50)

  if nb_step == 1

      X[j,1:n] = sol[1:n,1]'
      Y[j,1:n] = sol[1:n,2]'
      X[j,n+1] = cos(p[3]*sol.t[1]')
      Y[j,n+1] = cos(p[3]*sol.t[1]')

  else

      X[(j-1)*nb_step+1 : j*nb_step, 1:n] = sol[1:n,1:end-1]'
      Y[(j-1)*nb_step+1 : j*nb_step, 1:n] = sol[1:n,2:end]'
      X[(j-1)*nb_step+1 : j*nb_step, n_tot] = cos.(p[3]*sol.t[1:end-1]')
      Y[(j-1)*nb_step+1 : j*nb_step, n_tot] = cos.(p[3]*sol.t[1:end-1]')

  end

end

X[:,1:n] = X[:,1:n] + sigma_noise.*randn(size(X[:,1:n])).*X[:,1:n]
Y[:,1:n] = Y[:,1:n] + sigma_noise.*randn(size(Y[:,1:n])).*Y[:,1:n]

# identification

coeff, index = lifting_ident_main(X, Y, deg_F, time_step, deg_basis = deg_F, factor = 1., nb_inputs = 1)
nb_dic_F = size(index, 1)

# verification
coeff_exact = zeros(nb_dic_F, n)
for dim = 1 : n, j = 1 : size(F_coeff[dim], 1)
    ss = convert(Int,sum(F_coeff[dim][j, 1:n_tot]))
    if ss <= deg_F
        if ss > 0
            index_min = convert(Int,factorial(n_tot+ss-1)/factorial(ss-1)/factorial(n_tot)+1)
            index_max = convert(Int,factorial(n_tot+ss)/factorial(ss)/factorial(n_tot))
        else
            index_min = 1
            index_max = 1
        end
        val = falses(nb_dic_F)
        val[index_min:index_max] = [index[i,:]==F_coeff[dim][j,1:n_tot] for i = index_min:index_max]
        coeff_exact[val,dim] .= F_coeff[dim][j,n_tot+1]
    end
end

 # find(abs.(coeff_exact).>0)
index_non_zero = (LinearIndices(abs.(coeff_exact).>0))[findall(abs.(coeff_exact).>0)]

# figure
using Plots
pyplot()
plt = plot(index_non_zero, coeff_exact[index_non_zero], seriestype=:scatter, markershape = :circle, markersize = 20, msw = 3, markercolor = :white, msc = :blue, label = "exact")
plt = plot!(1:n*nb_dic_F, coeff[:], markershape = :circle, lw = 2, color = :red, markersize = 5, markercolor = :red, markerstrokewidths = 5, msc = :red, label = "estimated", legend = :best, xlabel = "coefficients index", ylabel = "coefficients values", tickfontsize = 20, legendfontsize = 14, guidefont = 20)
display(plt)

using Statistics
RMSE = norm(coeff-coeff_exact)/sqrt(n*nb_dic_F)
NRMSE = RMSE/mean(abs.(coeff_exact[index_non_zero]))

println("RMSE: ", RMSE)
println("NRMSE: ", NRMSE)
