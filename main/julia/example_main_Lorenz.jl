# Identification of the Lorenz system (main method)
# \dot{x} = 10(y-x); \dot{y} = x(28-z)-y; \dot{z} = xy-8/3 z

using DifferentialEquations
using LinearAlgebra
using SpecialFunctions

# dynamics
n = 3
p = [10.0,28.0,8/3]
function dyn(du,u,p,t)
du[1] = p[1]*(u[2]-u[1])
du[2] = u[1]*(p[2]-u[3]) - u[2]
du[3] = u[1]*u[2] - p[3]*u[3]
end
F_coeff = [ [0 1 0 p[1];1 0 0 -p[1]],
 [1 0 0 p[2];1 0 1 -1;0 1 0 -1.],
 [1 1 0 1.;0 0 1 -p[3]] ]

# parameters
deg_F = 3
t_end = 0.5
nb_step = 15
nb_simus = 20
ic_range = (-20.,20.)
sigma_noise = 0.01

# simulation
time_step = t_end / nb_step
nb_samples = nb_simus * nb_step
tspan = (0.0, t_end)
u0 = ic_range[1] .+ rand(nb_simus,n) .* (ic_range[2]-ic_range[1])

X = zeros(nb_samples, n)
Y = zeros(nb_samples, n)
for j in 1 : nb_simus

  prob = ODEProblem(dyn,u0[j,:],tspan,p)
  sol = solve(prob,Tsit5(),saveat=time_step,reltol=1e-9,abstol=1e-50)

  if nb_step == 1

      X[j,:] = sol[1:n,1]'
      Y[j, :] = sol[1:n,2]'

  else

      X[(j-1)*nb_step+1 : j*nb_step, :] = sol[1:n,1:end-1]'
      Y[(j-1)*nb_step+1 : j*nb_step, :] = sol[1:n,2:end]'

  end

end

X = X + sigma_noise.*randn(size(X)).*X
Y=Y + sigma_noise.*randn(size(Y)).*Y

# identification

coeff, index = lifting_ident_main(X, Y, deg_F, time_step, factor = 20.)
nb_dic_F = size(coeff, 1)

# verification
coeff_exact = zeros(nb_dic_F, n)
for dim = 1 : n, j = 1 : size(F_coeff[dim], 1)
    ss = convert(Int,sum(F_coeff[dim][j, 1:n]))
    if ss <= deg_F
        if ss > 0
            index_min = convert(Int,factorial(n+ss-1)/factorial(ss-1)/factorial(n)+1)
            index_max = convert(Int,factorial(n+ss)/factorial(ss)/factorial(n))
        else
            index_min = 1
            index_max = 1
        end
        val = falses(nb_dic_F)
        val[index_min:index_max] = [index[i,:]==F_coeff[dim][j,1:n] for i = index_min:index_max]
        coeff_exact[val,dim] .= F_coeff[dim][j,n+1]
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
