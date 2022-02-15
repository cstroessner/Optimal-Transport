% randomized SVD of A with rank r

function [US,V] = randSVD(A, rank, buffersize, iters)
if nargin < 3
    buffersize = 3; 
end
if nargin < 4
    iters = 1; 
end

[~,m] = size(A);

Q = rand([m,rank+buffersize]);
for i = 1:iters 
Q = A*Q;
[Q,~] = qr(Q,0);
end

[U,S,V] = svd(Q'*A);
U = Q*U;

U=U(:,1:rank);
V=V(:,1:rank);
S=S(1:rank,1:rank);
US = U*S; V = V';
end