function [r1,r2,r3,r4,r5] = MMSinkhorn5GaussianBridge(U,V,a,b,tol)

% set default parameters if not given
maxIters = 2000;

% initialize
n = size(U,1);
x1 = zeros([n,1]);
x2 = zeros([n,1]);
x3 = zeros([n,1]);
x4 = zeros([n,1]);
x5 = zeros([n,1]);

% we do not need to compute the normalization used by Friedland
% a normalization happens automatically after the first iteration

usedIters = -1;
success = 0;


% Sinkhorn iterations
for iter = 1:maxIters
    [s1,s2,s3,s4,s5] = computeMarginals(U,V,x1,x2,x3,x4,x5);
    n1 = norm(s1-s1'*a/(norm(a,2)^2).*a,1);
    n5 = norm(s5-s5'*b/(norm(b,2)^2).*b,1);
    err(iter) = max([n1,n5]);
    if err(iter) < tol/10   %Stopping Criterion 3.16
        usedIters = iter-1;
        success = 1;
        fprintf("Friedland sucessfull. ")
        break
    end
    I = find([n1,n5] == max([n1,n5])); %Index 3.10
    I = I(1);
    if I == 1
        x1 = x1 + log(a) - log(s1); %3.11
    elseif I == 2
        x5 = x5 + log(b) - log(s5);
    end
    if (max(isnan([s1,s2,s3,s4,s5])))
        fprintf("Warning: Friedland leads to NaN. ")
        usedIters = iter;
        break
    end
end

% print notifications
if (usedIters == -1)
    fprintf("Warning: Friedland reached MaxIters. ")
    usedIters = maxIters;
end

[r1,r2,r3,r4,r5] = computeMarginals(U,V,x1,x2,x3,x4,x5);

end

function [r1,r2,r3,r4,r5] = computeMarginals(U,V,x1,x2,x3,x4,x5)
x1 = exp(x1); x2 = exp(x2); x3 =exp(x3); x4 = exp(x4); x5 = exp(x5);

L1 = V'*(U'*x1);
L2 = V'*(U'*(L1.*x2));
L3 = V'*(U'*(L2.*x3));
L4 = V'*(U'*(L3.*x4));

R1 = U*(V*x5);
R2 = U*(V*(R1.*x4));
R3 = U*(V*(R2.*x3));
R4 = U*(V*(R3.*x2));

r1 = abs(R4.*x1);
r2 = abs(L1.*R3.*x2);
r3 = abs(L2.*R2.*x3);
r4 = abs(L3.*R1.*x4);
r5 = abs(L4.*x5);

%% for debugging only
% N = numel(x1);
% full = zeros(N,N,N,N,N);
% for i = 1:N
%     for j = 1:N
%         for k = 1:N
%             for l = 1:N
%                 for m = 1:N
%                     full(i,j,k,l,m) = K(i,j)*K(j,k)*K(k,l)*K(l,m)*x1(i)*x2(j)*x3(k)*x4(l)*x5(m);
%                 end
%             end
%         end
%     end
% end
% rr1 =  sum(full,[2,3,4,5]);
% err = norm(r1-rr1)
end