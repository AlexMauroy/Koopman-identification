function [coeff, index, F_samples, K] = lifting_ident(X,Y,deg_F,time_step,varargin)

%  coeff = lifting_ident(X,Y,deg_F,time_step,varargin)
%
%  LIFTING_IDENT uses the lifting method to identify the coefficients of a
%  polynomial vector field
%
%  INPUTS: 
%  X,Y: data matrices (m by n, where m is the number of snapshots and n is
%  the number of states); inputs (if any) are assumed to be the last states
%  deg_F: total degree of the vector field
%  time_step: sampling time
%
%  OPTIONS:
% 'factor': scaling factor of the data (default: 1.)
% 'vector_field': if 'vector_field'=1, the function return the values of the vector field in F (default: 0)
% 'inputs' : number of inputs (default: 0)
% 'deg_basis' : total degree of monomials basis functions; this value should be greater or equal to deg_F (default: deg_basis = deg_F)
% 'add_basis' : (vector-valued) function that returns the value of additional basis functions for a given state (inputs and outputs are row vectors) (default: add_basis = @(x) [])
% Note that the 'factor' is set to 1 when basis functions are added
%  
% 
%  OUTPUTS:
%  coeff: matrix containing the coefficients of the vector field (one component of the vector field in each column)
%  index: matrix containing in each row the powers of the corresponding monomial
%  F_samples: matrix containing in each row the values of the vector field at each sample point (only if 'vector_field'==1)

% default values
factor = 1;
v_field = 0;
nb_inputs = 0;
deg_basis = deg_F;
add_basis = @(x) [];

if ~isempty(varargin)
    
    for k = 1:2:length(varargin)-1
        
        if strcmp(varargin{k},'factor')
            
            factor = varargin{k+1};
            
        elseif strcmp(varargin{k},'vector_field')
            
            v_field = varargin{k+1};
            
        elseif strcmp(varargin{k},'inputs')
            
            nb_inputs = varargin{k+1};
            
        elseif strcmp(varargin{k},'deg_basis')
            
            deg_basis = varargin{k+1};
            
        elseif strcmp(varargin{k},'add_basis')
            
            add_basis = varargin{k+1};
            
        end
        
    end
    
end

n = size(X,2);
n_syst = n - nb_inputs;
nb_samples = size(X,1);


nb_dic_add = size(add_basis(X(1,:)),1)*size(add_basis(X(1,:)),2);

if nb_dic_add > 0
    factor = 1;
end

X = X/factor;
Y = Y/factor;


%% Construction of the Koopman matrix

% higher degree of the monomials (in x_k)
nb_dic = prod(deg_basis+1:n+deg_basis)/factorial(n);
nb_dic_tot = nb_dic + nb_dic_add;
nb_dic_F = prod(deg_F+1:n+deg_F)/factorial(n);

if nb_samples < nb_dic_tot
    
    error('Not enough samples: %d samples and %d basis functions', nb_samples, nb_dic_tot);
    return
    
end

% construct basis of monomials
index = [zeros(1,n);eye(n,n)];

for dim = 1 : n
    
    basis{dim} = index(dim+1,:);
    stack{dim} = basis{dim};
    
end

for k = 2 : deg_basis
    
    for dim = 1 : n
        
        current_stack = [];
        for dim2 = dim : n
            
            current_stack = [current_stack;stack{dim2}+ones(size(stack{dim2},1),1)*basis{dim}];
            
        end
        
        stack{dim} = current_stack;
        index = [index;current_stack];
        
    end
       
end

clear basis stack current_stack
   
psi_Y = zeros(nb_samples,nb_dic_tot);
psi_X = zeros(nb_samples,nb_dic_tot);

for k = 1 : nb_dic
    
    psi_X(:,k) = prod((X).^(ones(nb_samples,1)*index(k,:)),2);
    psi_Y(:,k) = prod((Y).^(ones(nb_samples,1)*index(k,:)),2);
    
end

if nb_dic_add > 0

    for j = 1 : nb_samples

        psi_X(j,nb_dic+1:nb_dic_tot) = add_basis(X(j,:));
        psi_Y(j,nb_dic+1:nb_dic_tot) = add_basis(Y(j,:));

    end

end

K = psi_X\psi_Y;
K(abs(K)<1e-6) = 0;


L = logm(K)./time_step;
L(abs(L)<1e-6) = 0;

%% find vector field values

if v_field == 1

    F_samples = psi_X*L(:,2:n+1);
    
else
    
    F_samples = [];
    
end

%% find coefficients

coeff = zeros(nb_dic_F+nb_dic_add,n_syst);
for dim = 1 : n_syst

    for k = 1 : nb_dic_F

        coeff(k,dim) = L(k,1+dim);

    end
    
    for k = 1 : nb_dic_add

        coeff(k+nb_dic_F,dim) = L(k+nb_dic,1+dim);

    end

end

% correction scaling

for k = 1 : nb_dic_F
    
    deg = sum(index(k,:));
    coeff(k,:) = coeff(k,:)*factor^(1-deg);
    
end

F_samples = F_samples*factor;
