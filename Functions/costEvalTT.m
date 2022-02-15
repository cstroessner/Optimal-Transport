% efficiently evaluates the cost term for TT approximation

function c = costEvalTT(G1,G2,G3,G4,x1,x2,x3,x4,y1,y2,y3,y4,C12,C23,C34)

% scaled term via network contractions
x1 = exp(x1); x2 = exp(x2); x3 =exp(x3); x4 = exp(x4);

% only works for same ranks and sizes
n = size(G1,2);
r = size(G1,3);

L1 = permute(G1,[3,2,1])*x1;
L2 = squeeze(sum(G2.* reshape(repmat(L1,[n,r]),[r,n,r]) .* permute(reshape(repmat(x2,[1,r*r]),[n,r,r]),[2,1,3]), [1,2]));

R1 = squeeze(G4)*x4;
R2 = squeeze(sum(G3.* permute(reshape(repmat(R1,[n,r]),[r,n,r]),[3,2,1]) .* permute(reshape(repmat(x3,[1,r*r]),[n,r,r]),[2,1,3]), [2,3]));

T12 = squeeze(sum(G2.* permute(reshape(repmat(R2,[n,r]),[r,n,r]),[3,2,1]),[3]));
T12 = squeeze(sum(reshape(repmat(reshape(G1,[n*r,1]),[1,n]),[n,r,n]) .* permute(reshape(repmat(T12',[1,n]),[n,r,n]),[3,2,1]),[2]));

T23r = squeeze(sum(G3.* permute(reshape(repmat(R1,[n,r]),[r,n,r]),[3,2,1]),[3]));
T23l = squeeze(sum(G2.* reshape(repmat(L1,[n,r]),[r,n,r]),[1]));
T23 = squeeze(sum(reshape(repmat(T23l,[1,n]),[n,r,n]) .* permute(reshape(repmat(T23r',[1,n]),[n,r,n]),[3,2,1]),[2]));

T34 = squeeze(sum(G3.* reshape(repmat(L2,[n,r]),[r,n,r]),[1]));
T34 = squeeze(sum(reshape(repmat(T34,[1,n]),[n,r,n]) .* permute(reshape(repmat(G4',[1,n]),[n,r,n]),[3,2,1]) ,[2]));

cS = sum(((C12 * diag(x2))'*diag(x1))'.* T12 ,[1,2]) + ...
    sum((diag(x2) * C23 * diag(x3)).* T23,[1,2]) + ...
    sum(((C34 * diag(x4))'*diag(x3))'.* T34,[1,2]);

% rank 1 term
cR1 = (C12*y2)'*y1 * sum(y3) * sum(y4) + ...
    (C23*y3)'*y2 * sum(y4) * sum(y1) + ...
    (C34*y4)'*y3 * sum(y1) * sum(y2);

c = cS + cR1;

end