%% Code to reproduce the experiment in the section generalized color transfer 
% YOU NEED TO MANUALLY PROVIDE JPG FILES AS SPECIFIED IN getPictures.m
addpath('Functions')
getPictures();

clear
close all
rng(1)
load('pictures.mat')
% contains 5 pictures with nxn pixels each stored in terms of vectors in R^{n^2 x 3}
% can be ploted using imshow(reshape(uint8(X1*n^2),n,n,3))
P1 = X1; P2 = X2; P3 = X3; P4 = X4; eta = 0.1;

%% plot for different ranks
lambda = [1,1,1];
Pr50 = colorTransfer(P1,P2,P3,P4,lambda,eta,50);
Pr3 = colorTransfer(P1,P2,P3,P4,lambda,eta,3);
Pr4 = colorTransfer(P1,P2,P3,P4,lambda,eta,5);
Pr5 = colorTransfer(P1,P2,P3,P4,lambda,eta,5);
Pr10 = colorTransfer(P1,P2,P3,P4,lambda,eta,10);
Pr15 = colorTransfer(P1,P2,P3,P4,lambda,eta,15);
Pr20 = colorTransfer(P1,P2,P3,P4,lambda,eta,20);
Pr30 = colorTransfer(P1,P2,P3,P4,lambda,eta,30);
Pinf = colorTransferFull(P1,P2,P3,P4,lambda, eta);

%%
figure(1)
montage({reshape(uint8(Pr3*256),n,n,3),reshape(uint8(Pr5*256),n,n,3),reshape(uint8(Pr10*256),n,n,3),reshape(uint8(Pr50*256),n,n,3),reshape(uint8(Pinf*256),n,n,3)},'size',[1 NaN])
%print -depsc 'figures/PicturesBarycenterRanks'

%% error plot for different ranks
errRanks(1) = norm(Pr3-Pinf,'inf');
errRanks(2) = norm(Pr5-Pinf,'inf');
errRanks(3) = norm(Pr10-Pinf,'inf');
errRanks(4) = norm(Pr15-Pinf,'inf');
errRanks(5) = norm(Pr20-Pinf,'inf');
errRanks(6) = norm(Pr30-Pinf,'inf');
errRanks(7) = norm(Pr50-Pinf,'inf');

figure(2)
set(gca,'fontsize',10)
set(figure(2), 'Position', [0 0 150 150])
semilogy([3,5,10,15,20,30,50],errRanks)
xlabel('$r$','interpreter','latex')
ylabel('$|| \tilde{x}^{r}-\tilde{x}^{*}||_{\infty}$','interpreter','latex')
%print -depsc 'figures/PicturesBarycenterRanksDecay'

%% plot different lambda with rank 10
PA = colorTransfer(P1,P2,P3,P4,[1,0,0],eta,50);
PB = colorTransfer(P1,P2,P3,P4,[0,1,0],eta,50);
PC = colorTransfer(P1,P2,P3,P4,[0,0,1],eta,50);
PD = colorTransfer(P1,P2,P3,P4,[1,2,0],eta,50);
PE = colorTransfer(P1,P2,P3,P4,[1,1,3],eta,50);

figure(3)
montage({reshape(uint8(P1*256),n,n,3),reshape(uint8(P2*256),n,n,3),reshape(uint8(P3*256),n,n,3),...
    reshape(uint8(PA*256),n,n,3),reshape(uint8(PB*256),n,n,3),reshape(uint8(PC*256),n,n,3),...
    reshape(uint8(PD*256),n,n,3),reshape(uint8(PE*256),n,n,3),reshape(uint8(P4*256),n,n,3)},'size',[3, 3])
%print -depsc 'figures/PicturesBarycenterLambdas'

%% runtime analysis
lambda = [1,1,1];
tic()
P = colorTransfer(P1,P2,P3,P4,lambda,eta,50);
time50 = toc();
tic()
P = colorTransferFull(P1,P2,P3,P4,lambda, eta);
timeFull = toc();
fprintf('Time for r=50 is %.2d. Time for full matrix is %.2d.\n',time50,timeFull)

%% functions
function P = colorTransfer(P1,P2,P3,P4,lambda, eta, r)
lambda = lambda/sum(lambda);
n = sqrt(size(P1,1));

%find Barycenter locations
PB = lambda(1)*P1+lambda(2)*P2+lambda(3)*P3;

% assemble pairwise cost matrices
C14 = zeros(n^2);
for i = 1:n^2
    for j = 1:n^2
        C14(i,j) = norm(P1(i,:)-PB(j,:))^2;
    end
end
C24 = zeros(n^2);
for i = 1:n^2
    for j = 1:n^2
        C24(i,j) = norm(P2(i,:)-PB(j,:))^2;
    end
end
C34 = zeros(n^2);
for i = 1:n^2
    for j = 1:n^2
        C34(i,j) = norm(P3(i,:)-PB(j,:))^2;
    end
end

% add weights
C14 = lambda(1)*C14;
C24 = lambda(2)*C24;
C34 = lambda(3)*C34;

% gibbs kernel
K14 = exp(-C14/eta);
K24 = exp(-C24/eta);
K34 = exp(-C34/eta);

% low-rank approximation via randomized SVD
[U14,V14] = randSVD(K14,r);
[U24,V24] = randSVD(K24,r);
[U34,V34] = randSVD(K34,r);

% marginals
a = ones(n^2,1);
b = ones(n^2,1);
c = ones(n^2,1);

% find density of Barycenterpoints
[d] = MMSinkhorn4BarycentersLineGraphLR(U14,V14,U24,V24,U34,V34, a, b, c, 1000,1e-4);

% get new vector
P = evaluateNewPixelValues(P4,PB,d,r,eta);
end

function [X] = evaluateNewPixelValues(P4,PB,d,r,eta)
n = sqrt(size(P4,1));

% assemble pairwise cost matrices
C = zeros(n^2);
for i = 1:n^2
    for j = 1:n^2
        C(i,j) = norm(PB(i,:)-P4(j,:))^2;
    end
end
K = exp(-C/eta);
[U,V] = randSVD(K,r);

% solve two marginal problem
[x1,x2] = MMSinkhorn2LR(U, V, d, ones(n^2,1), 1000, 1e-4);

x1 = exp(x1); x2 = exp(x2); 
X = real(((((PB.*x1)'*U)*V))'.*repmat(x2,1,3));
end

%% functions full
function P = colorTransferFull(P1,P2,P3,P4,lambda, eta)
lambda = lambda/sum(lambda);
n = sqrt(size(P1,1));

%find Barycenter locations
PB = lambda(1)*P1+lambda(2)*P2+lambda(3)*P3;

% assemble pairwise cost matrices
C14 = zeros(n^2);
for i = 1:n^2
    for j = 1:n^2
        C14(i,j) = norm(P1(i,:)-PB(j,:))^2;
    end
end
C24 = zeros(n^2);
for i = 1:n^2
    for j = 1:n^2
        C24(i,j) = norm(P2(i,:)-PB(j,:))^2;
    end
end
C34 = zeros(n^2);
for i = 1:n^2
    for j = 1:n^2
        C34(i,j) = norm(P3(i,:)-PB(j,:))^2;
    end
end

% add weights
C14 = lambda(1)*C14;
C24 = lambda(2)*C24;
C34 = lambda(3)*C34;

% gibbs kernel
K14 = exp(-C14/eta);
K24 = exp(-C24/eta);
K34 = exp(-C34/eta);

% low-rank approximation via randomized SVD
U14 = K14;
U24 = K24;
U34 = K34;
V14 = speye(size(U14));
V24 = speye(size(U24));
V34 = speye(size(U34));

% marginals
a = ones(n^2,1);
b = ones(n^2,1);
c = ones(n^2,1);

% find density of Barycenterpoints
[d] = MMSinkhorn4BarycentersLineGraphLR(U14,V14,U24,V24,U34,V34, a, b, c, 1000,1e-4);

% get new vector
P = evaluateNewPixelValuesFull(P4,PB,d,eta);
end

function [X] = evaluateNewPixelValuesFull(P4,PB,d,eta)
n = sqrt(size(P4,1));

% assemble pairwise cost matrices
C = zeros(n^2);
for i = 1:n^2
    for j = 1:n^2
        C(i,j) = norm(PB(i,:)-P4(j,:))^2;
    end
end
K = exp(-C/eta);
U = K;
V = speye(size(K));

% solve two marginal problem
[x1,x2] = MMSinkhorn2LR(U, V, d, ones(n^2,1), 1000, 1e-4);

x1 = exp(x1); x2 = exp(x2); 
X = real(((((PB.*x1)'*U)*V))'.*repmat(x2,1,3));
end



