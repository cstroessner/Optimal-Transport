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
Pr3 = colorTransfer(P1,P2,P3,P4,lambda,eta,3);
Pr4 = colorTransfer(P1,P2,P3,P4,lambda,eta,5);
Pr5 = colorTransfer(P1,P2,P3,P4,lambda,eta,5);
Pr10 = colorTransfer(P1,P2,P3,P4,lambda,eta,10);
Pr15 = colorTransfer(P1,P2,P3,P4,lambda,eta,15);
Pr20 = colorTransfer(P1,P2,P3,P4,lambda,eta,20);
Pr30 = colorTransfer(P1,P2,P3,P4,lambda,eta,30);
Pr50 = colorTransfer(P1,P2,P3,P4,lambda,eta,50);
Pinf = colorTransferFull(P1,P2,P3,P4,lambda, eta);

%%
figure(1)
montage({reshape(uint8(Pr3*256),n,n,3),reshape(uint8(Pr5*256),n,n,3),reshape(uint8(Pr10*256),n,n,3),reshape(uint8(Pr50*256),n,n,3),reshape(uint8(Pinf*256),n,n,3)},'size',[1 NaN])
print -depsc 'figures/PicturesRanks'

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
%print -depsc 'figures/PicturesRanksDecay'

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
%print -depsc 'figures/PicturesLambdas'

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

% assemble pairwise cost matrices
C14 = zeros(n^2);
for i = 1:n^2
    for j = 1:n^2
        C14(i,j) = norm(P1(i,:)-P4(j,:))^2;
    end
end
C24 = zeros(n^2);
for i = 1:n^2
    for j = 1:n^2
        C24(i,j) = norm(P2(i,:)-P4(j,:))^2;
    end
end
C34 = zeros(n^2);
for i = 1:n^2
    for j = 1:n^2
        C34(i,j) = norm(P3(i,:)-P4(j,:))^2;
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
d = ones(n^2,1);

% solve with MMSinkhorn
[x1, x2, x3, x4, err, usedIters, success] = MMSinkhorn4StarLR(U14,V14,U24,V24,U34,V34, a, b, c, d, 1000,1e-4);

% get new vector
P = evaluateNewPixelValues(U14,V14,U24,V24,U34,V34,x1,x2,x3,x4,P1,P2,P3,lambda);
end

function P = colorTransferFull(P1,P2,P3,P4,lambda, eta)
lambda = lambda/sum(lambda);
n = sqrt(size(P1,1));

% assemble pairwise cost matrices
C14 = zeros(n^2);
for i = 1:n^2
    for j = 1:n^2
        C14(i,j) = norm(P1(i,:)-P4(j,:))^2;
    end
end
C24 = zeros(n^2);
for i = 1:n^2
    for j = 1:n^2
        C24(i,j) = norm(P2(i,:)-P4(j,:))^2;
    end
end
C34 = zeros(n^2);
for i = 1:n^2
    for j = 1:n^2
        C34(i,j) = norm(P3(i,:)-P4(j,:))^2;
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

% use full tensors
U14 = K14; V14 = speye(n^2);
U24 = K24; V24 = speye(n^2);
U34 = K34; V34 = speye(n^2);

% marginals
a = ones(n^2,1);
b = ones(n^2,1);
c = ones(n^2,1);
d = ones(n^2,1);

% solve with MMSinkhorn
[x1, x2, x3, x4, err, usedIters, success] = MMSinkhorn4StarLR(U14,V14,U24,V24,U34,V34, a, b, c, d, 1000,1e-4);

% get new vector
P = evaluateNewPixelValues(U14,V14,U24,V24,U34,V34,x1,x2,x3,x4,P1,P2,P3,lambda);
end


function [X] = evaluateNewPixelValues(U14,V14,U24,V24,U34,V34,x1,x2,x3,x4,P1,P2,P3,lambda)
x1 = exp(x1); x2 = exp(x2); x3 =exp(x3); x4 = exp(x4);

K14 = U14*V14;
K24 = U24*V24;
K34 = U34*V34;

V1 = x1'*K14;
V2 = x2'*K24;
V3 = x3'*K34;
V1 = repmat(V1,3,1);
V2 = repmat(V2,3,1);
V3 = repmat(V3,3,1);
V4 = repmat(x4,1,3);

X = (((lambda(1)*P1.*x1)'*K14).*V2.*V3)'.*V4;
X = X + (((lambda(2)*P2.*x2)'*K24).*V1.*V3)'.*V4;
X = X + (((lambda(3)*P3.*x3)'*K34).*V1.*V2)'.*V4;
X = real(X);
end


