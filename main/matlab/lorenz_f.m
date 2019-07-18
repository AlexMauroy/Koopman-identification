function xdot = lorenz_f(t,x,param)


sigma = param.sigma;
rho = param.rho;
beta = param.beta;

if size( x, 3) > 1
    x = x(:);
end

nPoints = length( x) / 3;
ix = (0 * nPoints + 1) : (1 * nPoints);
iy = (1 * nPoints + 1) : (2 * nPoints);
iz = (2 * nPoints + 1) : (3 * nPoints);

xdot = [sigma*(x(iy)-x(ix));x(ix).*(rho-x(iz))-x(iy);x(ix).*x(iy)-beta*x(iz)];