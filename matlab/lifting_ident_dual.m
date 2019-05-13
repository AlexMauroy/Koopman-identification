function F_samples = lifting_ident_dual(X,Y,time_step,varargin)

%  coeff = lifting_ident_dual(X,Y,time_step,varargin)
%
%  LIFTING_IDENT_DUAL uses the lifting method to estimate the values of the
%  vector field at the data points
%
%  INPUTS: 
%  X,Y: data matrices (m by n, where m is the number of snapshots and n is the number of states)
%  time_step: sampling time
%
%  OPTIONS:
%  'basis': define the basis functions
%           'basis' = 0 : Gaussial radial basis functions
%           'basis' = n>0 : monomials of total degree less of equal to n (default: 3)
%  'factor': scaling factor of the data (default: 1)
%  'gamma': parameter of the radial basis functions (used if 'method' = 0) (default: 1/100)
% 
%  OUTPUTS:
%  F_samples: matrix containing in each row the values of the vector field at each sample point (only if 'vector_field'=1)

% default values
factor = 1;
meth = 3;
gamma_radial = 1/100;

if ~isempty(varargin)
    
    for k = 1:2:length(varargin)-1
        
        if strcmp(varargin{k},'basis')
            
            meth = varargin{k+1};
            
        elseif strcmp(varargin{k},'factor')
            
            factor = varargin{k+1};
            
        elseif strcmp(varargin{k},'gamma')
            
            gamma_radial = varargin{k+1};
            
        end
        
    end
    
end

n = size(X,2);
nb_samples = size(X,1);

X = X/factor;
Y = Y/factor;

%% Construction of the Koopman matrix

deg_basis = meth;
% higher degree of the monomials (in x_k)
nb_dic=round(factorial(n+deg_basis)/factorial(deg_basis)/factorial(n));

if and(nb_samples > nb_dic, meth ~= 0)
     
    error('Too many samples: %d samples and %d basis functions', nb_samples, nb_dic);
    
    return
    
end

if meth == 0 % basis of radial functions
    
    psi_Y = zeros(nb_samples);
    psi_X = zeros(nb_samples);

    for k = 1 : nb_samples

        psi_X(:,k) = exp(-sum((X-ones(nb_samples,1)*X(k,:)).^2,2)*gamma_radial);
        psi_Y(:,k) = exp(-sum((Y-ones(nb_samples,1)*X(k,:)).^2,2)*gamma_radial);

    end
    
    % %add identity functions
    % psi_X(:,nb_samples-n+1:nb_samples) = X
    % psi_Y(:,nb_samples-n+1:nb_samples) = Y
    
else % monomials
    
    index=[zeros(1,n);eye(n,n)];

    for dim=1:n

        basis{dim}=index(dim+1,:);
        stack{dim}=basis{dim};

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

    psi_Y = zeros(nb_samples,nb_dic);
    psi_X = zeros(nb_samples,nb_dic);
    for k = 1 : nb_dic

        psi_X(:,k) = prod((X).^(ones(nb_samples,1)*index(k,:)),2);
        psi_Y(:,k) = prod((Y).^(ones(nb_samples,1)*index(k,:)),2);

    end
    
end

K = psi_Y/psi_X;

L = logm(K)./time_step;
L(abs(L)<1e-6) = 0;
L = real(L);

%% find vector field values

F_samples = L*X;

F_samples = F_samples*factor;


