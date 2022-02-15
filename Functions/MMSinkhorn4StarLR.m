% Multimarginal Sinkhorn Algorithm for three marginals as in Friedland20. 

function [x1, x2, x3, x4, err, usedIters, success] = MMSinkhorn4StarLR(U14,V14,U24,V24,U34,V34, a, b, c, d, maxIters, tol, disableWarnings)

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
    
    %test only
%     x1 = rand(size(a));
%     x2 = rand(size(b));
%     x3 = rand(size(c));
%     x4 = rand(size(d));
    
    [s1,s2,s3,s4] = computeMarginals(U14,V14,U24,V24,U34,V34,x1,x2,x3,x4);
    
    % test only
%     K14 = U14*V14;    
%     K24 = U24*V24;
%     K34 = U34*V34;
%     Kfull = zeros(25,25,25,25);
%     for i = 1:25
%         for j = 1:25
%             for k = 1:25
%                 for l = 1:25
%                     Kfull(i,j,k,l) = K14(i,l)*K24(j,l)*K34(k,l)*x1(i)*x2(j)*x3(k)*x4(l);
%                 end
%             end
%         end
%     end
%     err1 = norm(s1-squeeze(sum(Kfull,[2,3,4])))
%     err2 = norm(s2'-squeeze(sum(Kfull,[1,3,4])))
%     err3 = norm(s3-squeeze(sum(Kfull,[1,2,4])))
%     err4 = norm(s4-squeeze(sum(Kfull,[1,2,3])))
    
    
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


function [s1,s2,s3,s4] = computeMarginals(U14,V14,U24,V24,U34,V34,x1,x2,x3,x4)
x1 = exp(x1); x2 = exp(x2); x3 =exp(x3); x4 = exp(x4);
v1 = x1'*U14;
v1 = v1*V14;
v2 = x2'*U24;
v2 = v2*V24;
v3 = x3'*U34;
v3 = v3*V34;
s4 = (v1 .* v2 .* v3 .* x4')';
s1 = V14*(v2.*v3.*x4')';
s1 = U14*s1;
s1 = s1.*x1;
s2 = V24*(v1.*v3.*x4')';
s2 = U24*s2;
s2 = s2.*x2;
s3 = V34*(v2.*v1.*x4')';
s3 = U34*s3;
s3 = s3.*x3;
end
