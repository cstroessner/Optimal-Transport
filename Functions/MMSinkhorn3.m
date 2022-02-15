% Multimarginal Sinkhorn Algorithm for three marginals as in Friedland20. 

function [P, err, usedIters, success] = MMSinkhorn3(C, a, b, c, lambda, maxIters, tol, disableWarnings)

% set default parameters if not given
if nargin<6
    maxIters = 5000;
end
if nargin<7
    tol = 1e-15;
end
if nargin<8
    disableWarnings = 0;
end
if (disableWarnings == 0)
    tic()
end

% initialize
n = size(C);
x1 = zeros([n(1),1]);
x2 = zeros([n(2),1]);
x3 = zeros([n(3),1]);
A = exp(-C./lambda);
A = A./sum(A,[1,2,3]);
usedIters = -1;
success = 0;


% Sinkhorn iterations
for iter = 1:maxIters
    P = computeP(x1,x2,x3,A); %Compute Tensor 3.9
    [s1,s2,s3] = computeMarginals(P);
    n1 = norm(s1-s1'*a/(norm(a,2)^2).*a,1);
    n2 = norm(s2-s2'*b/(norm(b,2)^2).*b,1);
    n3 = norm(s3-s3'*c/(norm(c,2)^2).*c,1);
    err(iter) = max([n1,n2,n3]);
    if err(iter) < tol/6   %Stopping Criterion 3.16
       usedIters = iter-1;
        success = 1;
        if disableWarnings == 0
            fprintf("Friedland sucessfull. ")
        end
        break 
    end
    I = find([n1,n2,n3] == max([n1,n2,n3])); %Index 3.10
    I = I(1);
    if I == 1
        x1 = x1 + log(a) - log(s1); %3.11
    elseif I == 2
        x2 = x2 + log(b) - log(s2);
    else %I==3
        x3 = x3 + log(c) - log(s3);
    end
    if (max(isnan(P)))
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

function P = computeP(b1,b2,b3,K)
n = size(K);
P = K;
for i = 1:n(1)
   for j = 1:n(2)
       for k = 1:n(3)
           P(i,j,k) = P(i,j,k) * exp(b1(i)+b2(j)+b3(k));
       end
   end  
end
end

function [r1,r2,r3] = computeMarginals(P)
r1 = squeeze(sum(P,[2,3]));
r2 = squeeze(sum(P,[1,3]));
r2 = r2';
r3 = squeeze(sum(P,[1,2]));
end
