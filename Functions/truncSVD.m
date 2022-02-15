% truncated SVD of M with rank r

function [UV,W] = truncSVD(M,r) %UV * W approx M with rank r
    [U,V,W] = svd(M);
    U = U(:,1:r); V = V(1:r,1:r); W = (W(:,1:r))';
    UV = U*V;
end
