% Identification of the four-states toggle switch system (main method)
% \dot{x} = -x + 2y
% \dot{y} = -y + 2/(1+z^2)
% \dot{z} = -2z + 2w
% \dot{w} = -2w + 1/(1+x)

clear all
close all

%%  data

% four state toggle switch
alpha = [2 1];
beta = [2 2];
gamma = [1 1 2 2];
n_exp = [2 1];

param_model.alpha = alpha;
param_model.beta = beta;
param_model.gamma = gamma;
param_model.n_exp = n_exp;
f_dyn     =   @(t,X) toggle_switch_f(t,X,param_model);

n = 4; % number of states

options = odeset('RelTol',1e-9,'AbsTol',1e-300);

% encoding of coefficients

% four state toggle switch
F_coeff{1}=[1 0 0 0 -gamma(1);0 1 0 0 beta(1)];
F_coeff{2}=[0 1 0 0 -gamma(2)];
F_coeff{3}=[0 0 1 0 -gamma(3);0 0 0 1 beta(2)];
F_coeff{4}=[0 0 0 1 -gamma(4)];
% additional terms
F_coeff_add{1}=[];
F_coeff_add{2}=[7 alpha(1)]; % each row contains the index of the basis function in the function "add_basis" and the associated coeffient
F_coeff_add{3}=[];
F_coeff_add{4}=[1 alpha(2)];

% simu parameters
t_end = 1;
nb_step = 2;
time_step = t_end/nb_step;
nb_simus = 100;

nb_samples = nb_simus*nb_step;

% initial conditions
range = [0 1];

X=[];
Y=[];
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
sigma_noise = 0.001;
X = X+sigma_noise*randn(size(X)).*X;
Y = Y+sigma_noise*randn(size(Y)).*Y;

%% identification

deg_F_tot = 1;
add_basis = @(x) [1/(1+x(1)) 1/(1+x(2)) 1/(1+x(3)) 1/(1+x(4)) 1/(1+x(1)^2) 1/(1+x(2)^2) 1/(1+x(3)^2) 1/(1+x(4)^2)];
[coeff, index, F_samples] = lifting_ident_main(X,Y,deg_F_tot,time_step,'vector_field',1,'factor',1,'add_basis',add_basis);
nb_dic_F = size(index, 1);  %number of monomial basis functions
nb_dic_add = length(add_basis(zeros(1,n))); %number of additional basis functions

%% verification

coeff_exact = [];
for dim = 1 : n

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

coeff_exact = reshape(coeff_exact, [nb_dic_F n]);

% additional terms
coeff_exact_add = zeros(nb_dic_add, n);
for dim = 1 : n
    for j = 1 : size(F_coeff_add{dim}, 1)

        coeff_exact_add(F_coeff_add{dim}(j,1),dim) = F_coeff_add{dim}(j,2);
        
    end
end
         
coeff_exact = [coeff_exact ; coeff_exact_add];
    
figure(1)
plot(1:n*(nb_dic_F+nb_dic_add),coeff(:),'rx','linewidth',2,'Markersize',8)
hold on
index_non_zero=find(abs(coeff_exact)>0);
plot(index_non_zero,coeff_exact(index_non_zero),'bo','linewidth',3,'Markersize',8)
plot([1 (nb_dic_F+nb_dic_add)*n],[0 0],'k--','linewidth',1)
xlabel('coefficients indices','Fontsize',12)
ylabel('coefficients values','Fontsize',12)
legend('estimated','exact')
box on
axis tight

RMSE = norm(coeff(:)-coeff_exact(:))/sqrt(n*(nb_dic_F+nb_dic_add));
NRMSE = RMSE/mean(abs(coeff_exact(index_non_zero)));