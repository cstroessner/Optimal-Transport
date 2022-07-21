

% Multimarginal Sinkhorn Algorithm for the Gaussian bridge setting with
% memory effect.

function [r1,r2,r3,r4,r5] = MMSinkhorn5GaussianBridgeMemoryEffect(U,V,a,b,tol)

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
    [iter,max([n1,n5])]
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
N = numel(x1); R = size(U,2);

R5 = zeros(R,R);
for i = 1:R
    for j = 1:R
        for k = 1:N
            R5(i,j) = R5(i,j) + x5(k)*V(i,k)*V(j,k);
        end
    end
end

R4 = zeros(R,R,R);
for i = 1:R
    for j = 1:R
        for k = 1:R
            for l = 1:N
                R4(i,j,k) = R4(i,j,k) + x4(l)*V(i,l)*V(j,l)*U(l,k);
            end
        end
    end
end

R45 = zeros(R,R,R);
for i = 1:R
    for j = 1:R
        for k = 1:R
            for l = 1:R
                R45(i,j,k) = R45(i,j,k) + R4(i,j,l)*R5(l,k);
            end
        end
    end
end

R3 = zeros(R,R,R,R);
for i = 1:R
    for j = 1:R
        for k = 1:R
            for l = 1:R
                for m = 1:N
                    R3(i,j,k,l) = R3(i,j,k,l) + x3(m)*V(i,m)*V(j,m)*U(m,k)*U(m,l);
                end
            end
        end
    end
end

R345 = zeros(R,R,R);
for i = 1:R
    for j = 1:R
        for k = 1:R
            for l = 1:R
                for m = 1:R
                    R345(i,j,k) = R345(i,j,k) + R3(i,j,l,m)*R45(l,k,m);
                end
            end
        end
    end
end

R2 = zeros(R,R,R,R);
for i = 1:R
    for j = 1:R
        for k = 1:R
            for m = 1:N
                R2(i,j,k) = R2(i,j,k) + x2(m)*V(i,m)*U(m,j)*U(m,k);
            end
        end
    end
end

R2345 = zeros(R,R);
for i = 1:R
    for j = 1:R
        for k = 1:R
            for l = 1:R
                R2345(i,j) = R2345(i,j) + R2(i,k,l)*R345(k,j,l);
            end
        end
    end
end

L1 = zeros(R,R);
for i = 1:R
    for j = 1:R
        for k = 1:N
            L1(i,j) = L1(i,j) + x1(k)*U(k,i)*U(k,j);
        end
    end
end

M1 = zeros(N,R,R);
for i = 1:R
    for j = 1:R
        for k = 1:N
            M1(k,i,j) = M1(k,i,j) + x1(k)*U(k,i)*U(k,j);
        end
    end
end

r1 = zeros(N,1);
for i = 1:N
    for j = 1:R
        for k = 1:R
            r1(i) = r1(i) + M1(i,j,k)*R2345(j,k);
        end
    end
end

M2 = zeros(N,R,R,R);
for i = 1:R
    for j = 1:R
        for k = 1:R
            for m = 1:N
                M2(m,i,j,k) = M2(m,i,j,k) + x2(m)*V(i,m)*U(m,j)*U(m,k);
            end
        end
    end
end

r2 = zeros(N,1);
for i = 1:N
    for j = 1:R
        for k = 1:R
            for l=1:R
                for m= 1:R
                    r2(i) = r2(i) + M2(i,j,k,l)*L1(j,m)*R345(k,m,l);
                end
            end
        end
    end
end

L12 = zeros(R,R,R);
for i = 1:R
    for j = 1:R
        for k = 1:R
            for l = 1:R
                L12(i,j,k) = L12(i,j,k) + L1(l,i)*R2(l,j,k);
            end
        end
    end
end

M3 = zeros(N,R,R,R,R);
for i = 1:R
    for j = 1:R
        for k = 1:R
            for l = 1:R
                for m = 1:N
                    M3(m,i,j,k,l) = M3(m,i,j,k,l) + x3(m)*V(i,m)*V(j,m)*U(m,k)*U(m,l);
                end
            end
        end
    end
end

Pre3 = zeros(R,R,R,R);
for j = 1:R
    for k = 1:R
        for l =1:R
            for m = 1:R
                for n = 1:R
                    Pre3(j,k,l,m) = Pre3(j,k,l,m) + L12(k,j,n)*R45(l,n,m);
                end
            end
        end
    end
end


r3 = zeros(N,1);
for i = 1:N
    for j = 1:R
        for k = 1:R
            for l =1:R
                for m = 1:R
                    r3(i) = r3(i) + M3(i,j,k,l,m)*Pre3(j,k,l,m);
                end
            end
        end
    end
end

L123 = zeros(R,R,R);
for i = 1:R
    for j = 1:R
        for k = 1:R
            for l = 1:R
                for m = 1:R
                    L123(i,j,k) = L123(i,j,k) + L12(l,m,i)*R3(m,l,j,k);
                end
            end
        end
    end
end

M4 = zeros(N,R,R,R);
for i = 1:R
    for j = 1:R
        for k = 1:R
            for l = 1:N
                M4(l,i,j,k) = M4(l,i,j,k) + x4(l)*V(i,l)*V(j,l)*U(l,k);
            end
        end
    end
end

r4 = zeros(N,1);
for i = 1:N
    for j = 1:R
        for k = 1:R
            for l =1:R
                for m = 1:R
                    r4(i) = r4(i) + M4(i,j,k,l)*L123(k,j,m)*R5(l,m);
                end
            end
        end
    end
end

L1234 = zeros(R,R);
for i = 1:R
    for j = 1:R
        for k = 1:R
            for l = 1:R
                L1234(i,j) = L1234(i,j) + L123(l,k,i)*R4(k,l,j);
            end
        end
    end
end

M5 = zeros(N,R,R);
for i = 1:R
    for j = 1:R
        for k = 1:N
            M5(k,i,j) = M5(k,i,j) + x5(k)*V(i,k)*V(j,k);
        end
    end
end

r5 = zeros(N,1);
for i = 1:N
    for j = 1:R
        for k = 1:R
            r5(i) = r5(i) + M5(i,j,k)*L1234(k,j);
        end
    end
end

% %% debug only
% K = U*V;
% full = zeros(N,N,N,N,N);
% % fullL1234 = zeros(R,R);
% % fullL123 = zeros(R,R,R);
% for i = 1:N
%     for j = 1:N
%         for k = 1:N
%             for l = 1:N
%                 for m = 1:N
%                     full(i,j,k,l,m) = K(i,j)*K(j,k)*K(k,l)*K(l,m)*K(i,k)*K(j,l)*K(k,m)*x1(i)*x2(j)*x3(k)*x4(l)*x5(m);
%                 end
% %                 for m= 1:R
% %                     for n = 1:R
% %                         %fullL1234(m,n) = fullL1234(m,n) + K(i,j)*K(j,k)*K(k,l)*U(l,n)*K(i,k)*K(j,l)*U(k,m)*x1(i)*x2(j)*x3(k)*x4(l);
% %                     end
% %                 end
%             end
% %             for l = 1:R
% %                 for m= 1:R
% %                     for n = 1:R
% %                         fullL123(l,m,n) = fullL123(l,m,n) + K(i,j)*K(j,k)*U(k,m)*U(k,n)*U(j,l)*K(i,k)*x1(i)*x2(j)*x3(k);
% %                     end
% %                 end
% %             end
%         end
%     end
% end
% rr1 =  sum(full,[2,3,4,5]);
% rr2 =  sum(full,[1,3,4,5])';
% rr3 =  squeeze(sum(full,[1,2,4,5]));
% rr4 =  squeeze(sum(full,[1,2,3,5]));
% rr5 =  squeeze(sum(full,[1,2,3,4]));
% 
% err1 = norm(r1-rr1)
% err2 = norm(r2-rr2)
% err3 = norm(r3-rr3)
% err4 = norm(r4-rr4)
% err5 = norm(r5-rr5)
% err = norm([r1,r2,r3,r4,r5]-[rr1,rr2,rr3,rr4,rr5])
end