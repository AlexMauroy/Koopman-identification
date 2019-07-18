% Identification of the Lorenz system (main method)
% \dot{x} = 10(y-x); \dot{y} = x(28-z)-y; \dot{z} = xy-8/3 z

clear all
close all

%% data

sigma = 10;
rho = 28;
beta = 8/3;

param_model.sigma = sigma;
param_model.rho = rho;
param_model.beta = beta;

f_dyn     =   @(t,X) lorenz_f(t,X,param_model);
n = 3; % number of states

options = odeset('RelTol',1e-9,'AbsTol',1e-300);

% encoding of coefficients (first three columns: degree in x1, x2, and u; fourth column: coefficient) 
F_coeff{1}=[0 1 0 sigma;1 0 0 -sigma];
F_coeff{2}=[1 0 0 rho;1 0 1 -1;0 1 0 -1];
F_coeff{3}=[1 1 0 1;0 0 1 -beta];

% simu parameters
t_end = 0.5;
nb_step = 15;
time_step = t_end/nb_step;
nb_simus = 20;

nb_samples = nb_simus*nb_step;

% initial conditions
range = [-20 20];

X = [];
Y = [];
for k = 1:nb_simus

    init_cond = range(1)*ones(1,n)+(range(2)-range(1))*rand(1,n);
    
    [t,x] = ode45(f_dyn,0:time_step:t_end,init_cond,options);
    
    if nb_step == 1
        
        X = [X;x(1,:)];
        Y = [Y;x(end,:)];
        
    else
        
        X = [X;x(1:end-1,:)];
        Y = [Y;x(2:end,:)];
        
    end
    
end

% measurement noise
sigma_noise = 0.01;
X = X+sigma_noise*randn(size(X)).*X;
Y = Y+sigma_noise*randn(size(Y)).*Y;

%% identification

deg_F_tot = 3;
[coeff, index, F_samples] = lifting_ident(X,Y,deg_F_tot,time_step,'vector_field',1,'factor',20);
nb_dic_F = size(index, 1);

%% verification

coeff_exact = [];
for dim = 1:n

    pol_F{dim} = zeros(1,nb_dic_F);
    for j = 1:size(F_coeff{dim},1)
        ss = sum(F_coeff{dim}(j,1:n));
        if ss > 0
            index_min = factorial(n+ss-1)/factorial(ss-1)/factorial(n)+1;
            index_max = factorial(n+ss)/factorial(ss)/factorial(n);
        else
            index_min = 1;
            index_max = 1;
        end
       pol_F{dim}([false(index_min-1,1);ismember(index(index_min:index_max,:),F_coeff{dim}(j,1:n),'rows')]) = F_coeff{dim}(j,n+1);
    end
    
    coeff_exact = [coeff_exact pol_F{dim}];
    
end

coeff_exact = reshape(coeff_exact, size(coeff));
    

figure(1)
plot(1:n*nb_dic_F,coeff(:),'rx','linewidth',2,'Markersize',8)
hold on
index_non_zero=find(abs(coeff_exact)>0);
plot(index_non_zero,coeff_exact(index_non_zero),'bo','linewidth',3,'Markersize',8)
plot([1 nb_dic_F*n],[0 0],'k--','linewidth',1)
legend('estimated','exact')
xlabel('coefficients indices','Fontsize',12)
ylabel('coefficients values','Fontsize',12)
box on
axis tight

RMSE = norm(coeff(:)-coeff_exact(:))/sqrt(n*nb_dic_F);
NRMSE = RMSE/mean(abs(coeff_exact(index_non_zero)));


