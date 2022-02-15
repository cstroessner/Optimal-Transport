% rounding of a tensor A of order 4 to satisfy marginals r1 r2 r3 r4 for
% Gibbs kernels given in TT format
% the output gives the updated scaling vectors x1,...,x4 and the additive
% rank-1 correction tensor y1 * y2 * y3 * y4 (where * is outer product)

function [x1,x2,x3,x4,y1,y2,y3,y4] = Rounding4TT(G1,G2,G3,G4,x1,x2,x3,x4,a,b,c,d)

x1 = exp(x1); x2 = exp(x2); x3 =exp(x3); x4 = exp(x4);

% only works for same ranks and sizes
n = size(G1,2);
r = size(G1,3);

R1 = squeeze(G4)*x4;
R2 = reshape(reshape(G3,[r*n,r])*R1,[r,n])*x3;
R3 = reshape(reshape(G2,[r*n,r])*R2,[r,n])*x2;

s1 = (squeeze(G1)*R3).*x1;
x1 = x1.*min(a./s1,ones(size(a)));

L1 = permute(G1,[3,2,1])*x1;
s2 = x2.*(reshape(L1'*reshape(G2,[r,n*r]),[n,r])*R2);
x2 = x2.*min(b./s2,ones(size(b)));

L2 = x2'*reshape(L1'*reshape(G2,[r,n*r]),[n,r]);
s3 = x3.*(reshape(L2*reshape(G3,[r,n*r]),[n,r])*R1);
x3 = x3.*min(c./s3,ones(size(c)));

L3 = x3'*reshape(L2*reshape(G3,[r,n*r]),[n,r]);
s4 = (L3*G4)'.*x4;
x4 = x4.*min(d./s4,ones(size(d)));

% rank 1 correction

R1 = squeeze(G4)*x4;
R2 = reshape(reshape(G3,[r*n,r])*R1,[r,n])*x3;
R3 = reshape(reshape(G2,[r*n,r])*R2,[r,n])*x2;

s1 = (squeeze(G1)*R3).*x1;
s2 = x2.*(reshape(L1'*reshape(G2,[r,n*r]),[n,r])*R2);
s3 = x3.*(reshape(L2*reshape(G3,[r,n*r]),[n,r])*R1);
s4 = (L3*G4)'.*x4;

y1 = a-s1;
y2 = b-s2;
y3 = c-s3;
y4 = d-s4;
y1 = y1./(norm(y1,1)^3);

% return log(scaling parameters) as used by Friedland
x1 = log(x1); x2 = log(x2); x3 = log(x3); x4 = log(x4);
end