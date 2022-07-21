%% Code to reproduce the experiment in the section proof of concept
clear, close all, rng(42)
addpath('Functions')
% REQUIRES MANUAL EXECUTION OF THE TT-TOOLBOX SETUP SCRIPT

n = 420; %number of points
k = 4; %number of marginals
d = 2; %dimension of points
rvec = 2:1:45;
rvecTT = 2:1:28;

% Compute Cost Matrices
X = rand(k,n,d);
C12 = zeros(n);
C23 = zeros(n);
C34 = zeros(n);

for a = 1:n
    for b = 1:n
        C12(a,b) = norm(squeeze(X(1,a,:))-squeeze(X(2,b,:))).^2;
        C23(a,b) = norm(squeeze(X(2,a,:))-squeeze(X(3,b,:))).^2;
        C34(a,b) = norm(squeeze(X(3,a,:))-squeeze(X(4,b,:))).^2;
    end
end
K12 = exp(-C12);
K23 = exp(-C23);
K34 = exp(-C34);

%% Analysis of Singular Values
svdKMats = [svd(K12),svd(K23),svd(K34)];

%% Transport Plan Based on the Full Tensor (exploiting the structure)
m = ones([n,1])./n;
for timeTest = 1:100
    tic();
    [x1,x2,x3,x4,~,iters] = MMSinkhorn4LineGraph(K12,K23,K34,m,m,m,m,1000,1e-4,1);
    [x1,x2,x3,x4,y1,y2,y3,y4] = Rounding4LineGraph(K12,K23,K34,x1,x2,x3,x4,m,m,m,m);
    % P(i,j,k,l) = K12(i,j)*K23(j,k)*K34(k,l)*exp(x1(i)+x2(j)+x3(k)+x4(l)) + y1(i)*y2(j)*y3(k)*y4(l);
    timeStructure = toc();
    timeStructures(timeTest) = timeStructure;
    costDirect = costEvalLineGraph(K12,K23,K34,x1,x2,x3,x4,y1,y2,y3,y4,C12,C23,C34);
end
timeStructure = mean(timeStructures);

%% Compute Approximations for SVDs
%precompute SVDs
[U12,S12,V12] = svd(K12);
[U23,S23,V23] = svd(K23);
[U34,S34,V34] = svd(K34);

% iteration over different ranks
costSVDs = zeros(size(rvec)); estLogErrSVDs = zeros(size(rvec)); timeSVDs = zeros(numel(rvec),100);
for i = 2:numel(rvec)
    r = rvec(i)
    
    % Approximation based on SVDs
    K12approx = U12(:,1:r)*S12(1:r,1:r)*V12(:,1:r)';
    K23approx = U23(:,1:r)*S23(1:r,1:r)*V23(:,1:r)';
    K34approx = U34(:,1:r)*S34(1:r,1:r)*V34(:,1:r)';
    
    % Compute transport plan and cost
    for timeTest = 1:100
        tic();
        [x1,x2,x3,x4,~,iterSVDs] = MMSinkhorn4LineGraphLR(U12(:,1:r),S12(1:r,1:r)*V12(:,1:r)',U23(:,1:r),S23(1:r,1:r)*V23(:,1:r)',U34(:,1:r),S34(1:r,1:r)*V34(:,1:r)',m,m,m,m,1000,1e-4,1);
        [x1,x2,x3,x4,y1,y2,y3,y4] = Rounding4LineGraphLR(U12(:,1:r),S12(1:r,1:r)*V12(:,1:r)',U23(:,1:r),S23(1:r,1:r)*V23(:,1:r)',U34(:,1:r),S34(1:r,1:r)*V34(:,1:r)',x1,x2,x3,x4,m,m,m,m);
        timeSVDs(i,timeTest) = toc();
        costSVDs(i) = costEvalLineGraphLR(U12(:,1:r),S12(1:r,1:r)*V12(:,1:r)',U23(:,1:r),S23(1:r,1:r)*V23(:,1:r)',U34(:,1:r),S34(1:r,1:r)*V34(:,1:r)',x1,x2,x3,x4,y1,y2,y3,y4,C12,C23,C34);
    end
    itersSVDs(i) = iterSVDs;
    estLogErr= 0;
    for samples = 1:1000
        a = randi(n,1); b = randi(n,1); c = randi(n,1); d = randi(n,1);
        val = K12(a,b)*K23(b,c)*K34(c,d);
        valSVDs = K12approx(a,b)*K23approx(b,c)*K34approx(c,d);
        estLogErr = max(estLogErr,abs(log(val)-log(valSVDs)));
    end
    estLogErrSVDs(i) = estLogErr;
    
    % evaluate the theoretical error bound for the entropic cost function
    elog = estLogErr; %only estimation available
    estop = 1e-4;
    Cnorm = max(max(abs(C12)+abs(C23)+abs(C34))); %only upper bound available
    boundSVDs(i) = 1*(elog*(2+log(2/elog))+elog/2*log(420^4-1)+2*estop*log(1/estop*(420^4-1)))+(elog+2*estop)*Cnorm;
end
timeSVDs = mean(timeSVDs,2)

%% Compute Approximations using the TT-SVD
costTT = zeros(size(rvecTT)); estLogErrTTsvd = zeros(size(rvecTT)); timeTT = zeros(numel(rvecTT),100);
K = @(I) K12(I(1),I(2)).*K23(I(2),I(3)).*K34(I(3),I(4));
TT=dmrg_cross(k,n,K,1e-12,'nswp',4,'rmin',30,'maxr',33);

%%
for i = 1:numel(rvecTT)
    r = rvec(i)
    
    TTtmp=round(TT,1e-16,r);
    G1 = core(TTtmp,1);
    G2 = core(TTtmp,2);
    G3 = core(TTtmp,3);
    G4 = core(TTtmp,4);
    
    for timeTest = 1:100
        tic();
        [x1,x2,x3,x4,~,iterTT] = MMSinkhorn4TT(G1,G2,G3,G4,m,m,m,m,1000,1e-4,1);
        [x1,x2,x3,x4,y1,y2,y3,y4] = Rounding4TT(G1,G2,G3,G4,x1,x2,x3,x4,m,m,m,m);
        timeTT(i,timeTest) = toc();
        %P(i,j,k,l) = (G1(1,i,:)'*G2(:,j,:)*G3(:,k,:)*G4(:,l,1)) *exp(x1(i)+x2(j)+x3(k)+x4(l)) + y1(i)*y2(j)*y3(k)*y4(l);
        costTT(i) = costEvalTT(G1,G2,G3,G4,x1,x2,x3,x4,y1,y2,y3,y4,C12,C23,C34);
    end
    itersTT(i) = iterTT;
    estLogErr= 0;
    for samples = 1:1000
        a = randi(n,1); b = randi(n,1); c = randi(n,1); d = randi(n,1);
        val = K12(a,b)*K23(b,c)*K34(c,d);
        valTTSVDs = squeeze(G1(1,a,:))'*squeeze(G2(:,b,:))*squeeze(G3(:,c,:))*squeeze(G4(:,d,1));
        estLogErr = max(estLogErr,abs(log(val)-log(valTTSVDs)));
    end
    estLogErrTTsvd(i) = estLogErr;
    
    % evalutate the theoretical error bound for the entropic cost function
    elog = estLogErr; %only estimation available
    estop = 1e-4;
    Cnorm = max(max(abs(C12)+abs(C23)+abs(C34))); %only upper bound available
    boundTT(i) = 1*(elog*(2+log(2/elog))+elog/2*log(420^4-1)+2*estop*log(1/estop*(420^4-1)))+(elog+2*estop)*Cnorm;
end
timeTT = mean(timeTT,2);


%% maybe only do runtime for a bigger example
close all
set(gca,'fontsize',10)
set(figure(1), 'Position', [0 0 400 350])
semilogy(rvecTT,abs(costTT-costDirect),'r'); hold on
semilogy(rvecTT,abs(costSVDs(1,1:numel(rvecTT))-costDirect),'b');
semilogy(rvecTT,estLogErrTTsvd,'r:');
semilogy(rvecTT,estLogErrSVDs(1:numel(rvecTT)),'b:');
%semilogy(rvecTT,boundSVDs(1:numel(rvecTT)),'r--');
%semilogy(rvecTT,boundTT(1:numel(rvecTT)),'b--');
leg = legend(...
    '$| \langle \mathcal{C} , \mathcal{P}_{\textsf{TT}} \rangle - \langle \mathcal{C} , \mathcal{P} \rangle  | $',...
    '$|\langle \mathcal{C} , \mathcal{P}_{\textsf{SVDs}} \rangle - \langle \mathcal{C} , \mathcal{P} \rangle  | $',...
    '$||\log(\mathcal{K}_{\textsf{TT}}) - \log(\mathcal{K}) ||_\infty$',...
    '$||\log(\mathcal{K}_{\textsf{SVDs}}) - \log(\mathcal{K}) ||_\infty$'...
);
set(leg,'Interpreter','latex');
xlabel('$r$','interpreter','latex')
xlim([2,rvecTT(end)])
%print -depsc 'figures/ProofOfConcept'


figure(2)
set(gca,'fontsize',10)
set(figure(2), 'Position', [0 0 400 350])
semilogy(2:rvec(end),ones(rvec(end)-1,1).*timeStructure,'k--')
hold on
semilogy(rvecTT,timeTT,'r')
semilogy(rvec,timeSVDs,'b')
leg = legend('$ \mathcal{P}$','$ \mathcal{P}_{\textsf{TT}}$','$ \mathcal{P}_{\textsf{SVDs}}$');
set(leg,'Interpreter','latex');
xlabel('$r$','interpreter','latex')
ylabel('time in seconds')
xlim([2,rvec(end)])
ylim([1.7e-3,7.3e-3])
%print -depsc 'figures/ProofTimes'

