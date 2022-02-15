% TT SVD very slow alternative to the TT toolbox for small n

function [G1,G2,G3,G4] = TTSVD(Tfun, n, rank)

% TT svd requires the full tensor
T = zeros([n,n,n,n]);
for i = 1:n
    for j = 1:n
        for k = 1:n
            for l = 1:n
                T(i,j,k,l) = Tfun(i,j,k,l);
            end
        end
    end
end

X = T;
r = [1, rank, rank, rank, 1];
ns = [n,n,n,n];

Ucore = cell(4);
for mu = 1:3
    X2 = reshape(X, r(mu) * ns(mu), prod(ns(mu+1:end)));
    [U, S, V] = svd(X2,0);
    
    U = U(:, 1:r(mu+1));
    S = S(1:r(mu+1), 1:r(mu+1));
    V=V(:,1:r(mu+1));
    
    Ucore{mu} = reshape(U, [r(mu), ns(mu), r(mu+1)]);
    X = reshape(S*V', [r(mu+1), ns(mu+1:end)]);
end
Ucore{4}=X;

G1 = Ucore{1};
G2 = Ucore{2};
G3 = Ucore{3};
G4 = Ucore{4};

end

