% Multimarginal Sinkhorn Algorithm for four marginals as in Friedland20.
% The Gibbs kernel needs to be provided as Line Graph in terms of low rank approximations K12 = U12 * V12

function [x1, x2, x3, x4, err, usedIters, success] = MMSinkhorn4LineGraphLR(U12,V12,U23,V23,U34,V34, a, b, c, d, maxIters, tol, disableWarnings)

% set default parameters if not given
if nargin<11
    maxIters = 5000;
end
if nargin<12
    tol = 1e-15;
end
if nargin<13
    disableWarnings = 0;
end
if (disableWarnings == 0)
    tic()
end

% initialize
x1 = zeros(size(a));
x2 = zeros(size(b));
x3 = zeros(size(c));
x4 = zeros(size(d));

% we do not need to compute the normalization used by Friedland
% a normalization happens automatically after the first iteration

usedIters = -1;
success = 0;

% Sinkhorn iterations
for iter = 1:maxIters
    [s1,s2,s3,s4] = computeMarginals(U12,V12,U23,V23,U34,V34,x1,x2,x3,x4);
    n1 = norm(s1-s1'*a/(norm(a,2)^2).*a,1);
    n2 = norm(s2-s2'*b/(norm(b,2)^2).*b,1);
    n3 = norm(s3-s3'*c/(norm(c,2)^2).*c,1);
    n4 = norm(s4-s4'*d/(norm(d,2)^2).*d,1);
    err(iter) = max([n1,n2,n3,n4]);
    if err(iter) < tol/8   %Stopping Criterion 3.16
        usedIters = iter-1;
        success = 1;
        if disableWarnings == 0
            fprintf("Friedland sucessfull. ")
        end
        break
    end
    I = find([n1,n2,n3,n4] == max([n1,n2,n3,n4])); %Index 3.10
    I = I(1);
    if I == 1
        x1 = x1 + log(a) - log(s1); %3.11
    elseif I == 2
        x2 = x2 + log(b) - log(s2);
    elseif I==3
        x3 = x3 + log(c) - log(s3);
    else
        x4 = x4 + log(d) - log(s4);
    end
    if max(isnan([x1;x2;x3;x4]),[],'all')
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


function [s1,s2,s3,s4] = computeMarginals(U12,V12,U23,V23,U34,V34,x1,x2,x3,x4)
x1 = exp(x1); x2 = exp(x2); x3 =exp(x3); x4 = exp(x4);

L1 = (V12'*(U12'*x1));
L2 = (V23'*(U23'*(L1.*x2)));
L3 = (V34'*(U34'*(L2.*x3)));

R1 = U34*(V34*x4);
R2 = U23*(V23*(R1.*x3));
R3 = U12*(V12*(R2.*x2));

s1 = R3.*x1;
s2 = L1.*R2.*x2;
s3 = L2.*R1.*x3;
s4 = L3.*x4;
end
