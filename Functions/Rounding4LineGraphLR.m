% rounding of a tensor A of order 4 to satisfy marginals r1 r2 r3 r4 for
% Gibbs kernels based on line graphs with low rank approximations
% the output gives the updated scaling vectors x1,...,x4 and the additive
% rank-1 correction tensor y1 * y2 * y3 * y4 (where * is outer product)

function [x1,x2,x3,x4,y1,y2,y3,y4] = Rounding4LineGraphLR(U12,V12,U23,V23,U34,V34,x1,x2,x3,x4,a,b,c,d)

x1 = exp(x1); x2 = exp(x2); x3 =exp(x3); x4 = exp(x4);

% update of the scaling parameters

R1 = U34*(V34*x4);
R2 = U23*(V23*(R1.*x3));
R3 = U12*(V12*(R2.*x2));

s1 = R3.*x1;
x1 = x1.*min(a./s1,ones(size(a)));

L1 = (V12'*(U12'*x1));
s2 = L1.*R2.*x2;
x2 = x2.*min(b./s2,ones(size(b)));

L2 = (V23'*(U23'*(L1.*x2)));
s3 = L2.*R1.*x3;
x3 = x3.*min(c./s3,ones(size(c)));

L3 = (V34'*(U34'*(L2.*x3)));
s4 = L3.*x4;
x4 = x4.*min(d./s4,ones(size(d)));

% rank 1 correction

R1 = U34*(V34*x4);
R2 = U23*(V23*(R1.*x3));
R3 = U12*(V12*(R2.*x2));

s1 = R3.*x1;
s2 = L1.*R2.*x2;
s3 = L2.*R1.*x3;
s4 = L3.*x4;

y1 = a-s1;
y2 = b-s2;
y3 = c-s3;
y4 = d-s4;
y1 = y1./(norm(y1,1)^3);


% return log(scaling parameters) as used by Friedland
x1 = log(x1); x2 = log(x2); x3 = log(x3); x4 = log(x4);
end