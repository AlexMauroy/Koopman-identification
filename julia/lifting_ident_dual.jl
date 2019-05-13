function lifting_ident_dual(X,Y,time_step; factor = 1., basis = 3, gamma = 1/100)

# F_samples = lifting_ident_dual(X,Y,time_step, optional arguments)
#
# LIFTING_IDENT_DUAL uses the lifting method to estimate the values of the vector field at the data points
#
# INPUTS:
# X,Y: data matrices (m by n, where m is the number of snapshots and n is the number of states)
# time_step: sampling time
#
# OPTIONS:
# 'basis': define the basis functions
#          'basis' = 0 : Gaussial radial basis functions
#          'basis' = n>0 : monomials of total degree less of equal to n (default: 3)
# 'factor': scaling factor of the data (default: 1.)
# 'gamma': parameter of the radial basis functions (used if 'method' = 0) (default: 1/100)
#
# OUTPUTS:
# F_samples: matrix containing in each row the values of the vector field at each sample point (only if 'vector_field'=1)

n = size(X,2)
nb_samples = size(X,1)

X = X/factor
Y = Y/factor

# Construction of the Koopman matrix

deg_basis = basis
# higher degree of the monomials (in x_k)
nb_dic = convert(Int, prod(n+1:n+deg_basis)/factorial(deg_basis))

if nb_samples > nb_dic && basis != 0

    println("Too many samples: ", nb_samples, " samples and ", nb_dic, " basis functions")

    return [], [], []

end

if basis == 0 #basis of radial functions

    psi_Y = zeros(nb_samples,nb_samples)
    psi_X = zeros(nb_samples,nb_samples)
    for k = 1 : nb_samples

        psi_X[:,k] = exp.(-sum((X-ones(nb_samples,1)*X[k,:]').^2,dims=2)*gamma)
        psi_Y[:,k] = exp.(-sum((Y-ones(nb_samples,1)*X[k,:]').^2,dims=2)*gamma)

    end

    # #add identity functions
    # psi_X[:,nb_samples-n+1:nb_samples] = X
    # psi_Y[:,nb_samples-n+1:nb_samples] = Y

else # monomials

    index = [zeros(Int,1,n);Matrix{Int}(I,n,n)]

    stack = Dict(dim => convert(Array{Int,2},hcat(index[dim+1,:])') for dim = 1 : n)

    for k = 2 : deg_basis, dim = 1 : n

        current_stack = Array{Int}(undef,0,n)
        for dim2 = dim : n

            current_stack = [current_stack;stack[dim2]+ones(Int,size(stack[dim2],1),1)*index[dim+1,:]']

        end

        stack[dim] = current_stack
        index = [index;current_stack]

    end

    stack = 0
    current_stack = 0

    psi_Y = zeros(nb_samples,nb_dic)
    psi_X = zeros(nb_samples,nb_dic)
    for k = 1 : nb_dic

        psi_X[:,k]=prod(X.^(ones(nb_samples,1)*index[k,:]'),dims=2)
        psi_Y[:,k]=prod(Y.^(ones(nb_samples,1)*index[k,:]'),dims=2)

    end

end

K = psi_Y/psi_X

L = log(K)./time_step
L[abs.(L).<1e-6] .= 0

L = real(L)

# find vector field values

F_samples = L*X
F_samples = F_samples*factor

return F_samples

end
