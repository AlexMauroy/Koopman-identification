function x_dot = duffing_f(t,x,param)

mu = param.mu;
gamma = param.gamma;
omega = param.omega;


if size( x, 2) > 1
    x = x(:);
end

nPoints = length( x) / 2;
ix = (0 * nPoints + 1) : (1 * nPoints);
iy = (1 * nPoints + 1) : (2 * nPoints);

x_dot = [x(iy);x(ix)-x(ix).^3-mu*x(iy)+x(ix).^2*gamma*cos(omega*t)];
