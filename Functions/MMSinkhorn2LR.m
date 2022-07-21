% Multimarginal Sinkhorn Algorithm for three marginals as in Friedland20. 

function [x1,x2] = MMSinkhorn2LR(U,V, a, b, maxIters, tol, disableWarnings)

% set default parameters if not given
if nargin<5
    maxIters = 5000;
end
if nargin<6
    tol = 1e-15;
end
if nargin<7
    disableWarnings = 0;
end
if (disableWarnings == 0)
    tic()
end

% initialize
n = [size(U,1),size(V,2)];
x1 = zeros([n(1),1]);
x2 = zeros([n(2),1]);

% we do not need to compute the normalization used by Friedland
% a normalization happens automatically after the first iteration

usedIters = -1;
success = 0;


% Sinkhorn iterations
for iter = 1:maxIters
    [s1,s2] = computeMarginals(U,V,x1,x2);
    n1 = norm(s1-s1'*a/(norm(a,2)^2).*a,1);
    n2 = norm(s2-s2'*b/(norm(b,2)^2).*b,1);
    err(iter) = max([n1,n2]);
    if err(iter) < tol/8   %Stopping Criterion 3.16
        usedIters = iter-1;
        success = 1;
        if disableWarnings == 0
            fprintf("Friedland sucessfull. ")
        end
        break
    end
    I = find([n1,n2] == max([n1,n2])); %Index 3.10
    I = I(1);
    if I == 1
        x1 = x1 + log(a) - log(s1); %3.11
    else %I == 2
        x2 = x2 + log(b) - log(s2);
    end
    if max(isnan([x1;x2]),[],'all')
        if (disableWarnings == 0)
            fprintf("Warning: Friedland leads to NaN. ")
        end
        usedIters = iter;
        break
    end
end

% print notifications
if (usedIters == -1)
    if (disableWarnings == 0)
        fprintf("Warning: Friedland reached MaxIters. ")
    end
    usedIters = maxIters;
end

if (disableWarnings == 0)
    toc()
end
end


function [s1,s2] = computeMarginals(U,V,x1,x2)
x1 = exp(x1); x2 = exp(x2); 

L1 = (V'*(U'*x1));
R1 = U*(V*x2);

s1 = R1.*x1;
s2 = L1.*x2;
end