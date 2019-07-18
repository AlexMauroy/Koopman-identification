function dX = nonlin_network_f(t,X,param)

F = param.F;
A_lin = param.A_lin;
n = size(F,2);

if size( X, 2) > 1
    X = X(:);
end

nPoints = length( X) / n;

for k = 1 : n


    ix{k} = ((k-1) * nPoints + 1) : (k * nPoints);

end

dX = kron(A_lin,eye(nPoints))*X;

for dim = 1 : n
    
    for k = 1 : size(F{dim},1)
    
        dX(ix{dim}) = dX(ix{dim})+F{dim}(k,5)*X(ix{F{dim}(k,1)}).^F{dim}(k,3).*X(ix{F{dim}(k,2)}).^F{dim}(k,4);

    end
    
end