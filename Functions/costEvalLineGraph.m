% efficiently evaluates the cost term for the line graph setting with full K12 K23 K34

function c = costEvalLineGraph(K12,K23,K34,x1,x2,x3,x4,y1,y2,y3,y4,C12,C23,C34)

% scaled term
x1 = exp(x1); x2 = exp(x2); x3 =exp(x3); x4 = exp(x4);

L1 = (K12'*x1);
L2 = (K23'*(L1.*x2));

R1 = K34*x4;
R2 = K23*(R1.*x3);

cS = sum(((C12 * diag(x2))'*diag(x1))'.*(K12*diag(R2)),[1,2]) + ...
    sum((diag(x2) * C23 * diag(x3)).*(diag(L1)*K23*diag(R1)),[1,2]) + ...
    sum(((C34 * diag(x4))'*diag(x3))'.*(diag(L2)*K34),[1,2]);

% rank 1 term
cR1 = (C12*y2)'*y1 * sum(y3) * sum(y4) + ...
    (C23*y3)'*y2 * sum(y4) * sum(y1) + ...
    (C34*y4)'*y3 * sum(y1) * sum(y2);

c = cS + cR1;

end