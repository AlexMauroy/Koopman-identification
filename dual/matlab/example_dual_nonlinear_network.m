% identification of the vector field values of a nonlinear networked system (dual method)

clear all
close all

%% data

% network with nonlinear couplings;
n = 30;
nb_interactions = 5;
power_max = 3;
amplitude_coeff = 1;
A_lin = 1*diag(-rand(1,n));
for dim = 1 : n
    F{dim} = [zeros(nb_interactions,4) amplitude_coeff*randn(nb_interactions,1)];
    for j = 1 : nb_interactions
        tot_deg = randi(power_max-1) + 1;
        first_deg = randi(tot_deg);
        F{dim}(j,1:4) = [randperm(n, 2) first_deg tot_deg-first_deg];
    end
end

param_model.F = F;
param_model.A_lin = A_lin;
f_dyn     =   @(t,X) nonlin_network_f(t,X,param_model);

options = odeset('RelTol',1e-9,'AbsTol',1e-300);

% simu parameters
t_end = 1;
nb_step = 2;
time_step = t_end/nb_step;
nb_simus = 200;
nb_samples = nb_simus * nb_step;

number_traject_per_loop = 50;
number_loops = ceil(nb_simus/number_traject_per_loop);

% initial conditions
range=0.5*[-1 1];

X = [];
Y = [];

for kk = 1:number_loops
    
    % setting the initial conditions
    index_low = 1+(kk-1)*number_traject_per_loop;
    index_high = min(kk*number_traject_per_loop,nb_simus);
    number_traj_loop = index_high-index_low+1;
    
    
    disp(['batch ' num2str(kk) ' / ' num2str(number_loops)])

    init_cond=range(1)*ones(1,n*number_traj_loop)+(range(2)-range(1))*rand(1,n*number_traj_loop);

    [t,x]=ode45(f_dyn,0:time_step:t_end,init_cond,options);

    
    if nb_step == 1
        
        X = [X;x(1,:)];
        Y = [Y;x(end,:)];
        
    else
        
        X = [X;x(1:end-1,:)];
        Y = [Y;x(2:end,:)];
        
    end
      
end

X = reshape(X,[nb_step*nb_simus,n]);
Y = reshape(Y,[nb_step*nb_simus,n]);

% measurement noise
sigma_noise=0.01;
X=X+sigma_noise*randn(size(X)).*X;
Y=Y+sigma_noise*randn(size(Y)).*Y;

%% identification

deg_basis = 0;

F_samples = lifting_ident_dual(X,Y,time_step,'basis',deg_basis,'gamma',0.01,'factor',1);

%% verification

F_exact = zeros(nb_samples, n);
for k = 1 : nb_samples
    
    F_exact(k,:) = f_dyn(0,X(k,:))';
    
end

RMSE = norm(F_samples(:)-F_exact(:))/sqrt(n*nb_samples);
NRMSE = RMSE/mean(abs(F_exact(:)));

figure(1)
quiver(X(:,1),X(:,2),F_samples(:,1),F_samples(:,2))
hold on
quiver(X(:,1),X(:,2),F_exact(:,1),F_exact(:,2),'r')
xlabel('x_1','fontsize',12)
ylabel('x_2','fontsize',12)
legend('estimated','exact')
