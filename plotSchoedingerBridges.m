clear
addpath('Functions')

n = 40; 
r = 10;
eta = 0.1;
tol = 1e-4;

% cost matrix
for i = 1:n
    for j = 1:n
        for k = 1:n
            for l = 1:n
                C(sub2ind([n,n],i,j),sub2ind([n,n],k,l)) =  norm([i/n,j/n]-[k/n,l/n])^2;
            end
        end
    end
end
K = exp(-C/eta);
% low-rank approximation
[U,V] = truncSVD(K,r);

% initial marginals
m1 = zeros(n,n);
m1(1:n/2,1:n/2) = 1; m1 = m1(:);
m1 = m1+0.001; m1 = m1./sum(m1);
m2 = zeros(n,n);
m2(n/2+1:end,n/2+1:end) = 1; m2 = m2(:);
m2 = m2 + 0.001; m2 = m2./sum(m2);

% no memory effect
[mm1,mm2,mm3,mm4,mm5] = MMSinkhorn5GaussianBridge(U,V,m1,m2,tol);

% with memory effect
[mmm1,mmm2,mmm3,mmm4,mmm5] = MMSinkhorn5GaussianBridgeMemoryEffect(U,V,m1,m2,tol);

%% plot
figure(1)
set(figure(1), 'Position', [0 0 700 250])
subplot(2,6,3)
imagesc(reshape(mm3,n,n))
set(gca,'XTick',[], 'YTick', [])
caxis([0 0.0027]);
xlabel('$r_3(\mathcal P)$','Interpreter','latex');
subplot(2,6,4)
imagesc(reshape(mm4,n,n))
set(gca,'XTick',[], 'YTick', [])
caxis([0 0.0027]);
xlabel('$r_4(\mathcal P)$','Interpreter','latex');
subplot(2,6,5)
imagesc(reshape(mm5,n,n))
set(gca,'XTick',[], 'YTick', [])
caxis([0 0.0027]);
xlabel('$r_5$','Interpreter','latex');
subplot(2,6,1)
imagesc(reshape(mm1,n,n))
set(gca,'XTick',[], 'YTick', [])
caxis([0 0.0027]);
xlabel('$r_1$','Interpreter','latex');
subplot(2,6,2)
imagesc(reshape(mm2,n,n))
set(gca,'XTick',[], 'YTick', [])
caxis([0 0.0027]);
xlabel('$r_2(\mathcal P)$','Interpreter','latex');
subplot(2,6,9)
imagesc(reshape(mmm3,n,n))
set(gca,'XTick',[], 'YTick', [])
caxis([0 0.0027]);
xlabel('$r_3(\mathcal P)$','Interpreter','latex');
subplot(2,6,10)
imagesc(reshape(mmm4,n,n))
set(gca,'XTick',[], 'YTick', [])
caxis([0 0.0027]);
xlabel('$r_4(\mathcal P)$','Interpreter','latex');
subplot(2,6,11)
imagesc(reshape(mmm5,n,n))
set(gca,'XTick',[], 'YTick', [])
caxis([0 0.0027]);
xlabel('$r_5$','Interpreter','latex');
subplot(2,6,7)
imagesc(reshape(mmm1,n,n))
set(gca,'XTick',[], 'YTick', [])
caxis([0 0.0027]);
xlabel('$r_1$','Interpreter','latex');
subplot(2,6,8)
imagesc(reshape(mmm2,n,n))
set(gca,'XTick',[], 'YTick', [])
caxis([0 0.0027]);
xlabel('$r_2(\mathcal P)$','Interpreter','latex');
subplot(2,6,12)
axis off
caxis([0 0.0027]);
colorbar()
%print -depsc 'figures/Schroedinger'




