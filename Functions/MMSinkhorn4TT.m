% Multimarginal Sinkhorn Algorithm for four marginals as in Friedland20.
% The Gibbs kernel needs to be provided as TT approximation with cores G1 G2 G3 G4

function [x1,x2,x3,x4, err, usedIters, success] = MMSinkhorn4TT(G1,G2,G3,G4, a, b, c, d, maxIters, tol, disableWarnings)

% set default parameters if not given
if nargin<9
    maxIters = 5000;
end
if nargin<10
    tol = 1e-15;
end
if nargin<11
    disableWarnings = 0;
end
if (disableWarnings == 0)
    tic()
end

% initialize
n = [size(G1,2),size(G2,2),size(G3,2),size(G4,2)];
x1 = zeros([n(1),1]);
x2 = zeros([n(2),1]);
x3 = zeros([n(3),1]);
x4 = zeros([n(4),1]);
usedIters = -1;
success = 0;


% Sinkhorn iterations
for iter = 1:maxIters
    [s1,s2,s3,s4] = computeMarginals(G1,G2,G3,G4,x1,x2,x3,x4); 
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

function [s1,s2,s3,s4] = computeMarginals(G1,G2,G3,G4,x1,x2,x3,x4)
x1 = exp(x1); x2 = exp(x2); x3 =exp(x3); x4 = exp(x4);

% only works for same ranks and sizes
n = size(G1,2);
r = size(G1,3);

L1 = permute(G1,[3,2,1])*x1;
L2 = x2'*reshape(L1'*reshape(G2,[r,n*r]),[n,r]);
L3 = x3'*reshape(L2*reshape(G3,[r,n*r]),[n,r]);

R1 = squeeze(G4)*x4;
R2 = reshape(reshape(G3,[r*n,r])*R1,[r,n])*x3;
R3 = reshape(reshape(G2,[r*n,r])*R2,[r,n])*x2;

s1 = (squeeze(G1)*R3).*x1;
s2 = x2.*(reshape(L1'*reshape(G2,[r,n*r]),[n,r])*R2);
s3 = x3.*(reshape(L2*reshape(G3,[r,n*r]),[n,r])*R1);
s4 = (L3*G4)'.*x4;
end