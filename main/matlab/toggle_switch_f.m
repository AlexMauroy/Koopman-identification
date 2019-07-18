function dX = toggle_switch_f(t,X,param)

alpha = param.alpha;
beta = param.beta;
gamma = param.gamma;
n_exp = param.n_exp;

if size( X, 2) > 1
    X = X(:);
end

nPoints = length( X) / 4;
ix = (0 * nPoints + 1) : (1 * nPoints);
iy = (1 * nPoints + 1) : (2 * nPoints);
iz = (2 * nPoints + 1) : (3 * nPoints);
iw = (3 * nPoints + 1) : (4 * nPoints);

dX = [beta(1)*X(iy)-gamma(1)*X(ix);
    -gamma(2)*X(iy)+alpha(1)./(1+X(iz).^n_exp(1));
    beta(2)*X(iw)-gamma(3)*X(iz);
    -gamma(4)*X(iw)+alpha(2)./(1+X(ix).^n_exp(2))];
