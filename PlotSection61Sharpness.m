%% Code to reproduce the experiment in the section proof of concept
clear, close all, rng(42)
addpath('Functions')
% REQUIRES MANUAL EXECUTION OF THE TT-TOOLBOX SETUP SCRIPT

n = 10; %number of points
k = 4; %number of marginals
d = 2; %dimension of points
rvec = 2:1:n;
rvecTT = 2:1:n;
estop = 1e-4;
eta = 1;

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
Cnorm = 0;
for a = 1:n
    for b = 1:n
        for c = 1:n
            for d = 1:n
                Cnorm = max(Cnorm,abs(C12(a,b)+C23(b,c)+C34(c,d)));
            end
        end
    end
end

%% Transport Plan Based on the Full Tensor (exploiting the structure)
m = ones([n,1])./n;
for timeTest = 1:100
    tic();
    [x1,x2,x3,x4,~,iters] = MMSinkhorn4LineGraph(K12,K23,K34,m,m,m,m,1000,estop,1);
    [x1r,x2r,x3r,x4r,y1,y2,y3,y4] = Rounding4LineGraph(K12,K23,K34,x1,x2,x3,x4,m,m,m,m);
    % P(i,j,k,l) = K12(i,j)*K23(j,k)*K34(k,l)*exp(x1(i)+x2(j)+x3(k)+x4(l)) + y1(i)*y2(j)*y3(k)*y4(l);
    timeStructure = toc();
    timeStructures(timeTest) = timeStructure;
    costDirect = costEvalLineGraph(K12,K23,K34,x1r,x2r,x3r,x4r,y1,y2,y3,y4,C12,C23,C34);
    costDirectNoRounding = costEvalLineGraph(K12,K23,K34,x1,x2,x3,x4,zeros(size(y1)),zeros(size(y2)),zeros(size(y3)),zeros(size(y4)),C12,C23,C34);
end
timeStructure = mean(timeStructures);

for i = 1:n
    for j = 1:n
        for k = 1:n
            for l = 1:n
                P(i,j,k,l) = K12(i,j)*K23(j,k)*K34(k,l)*exp(x1(i)+x2(j)+x3(k)+x4(l));
            end
        end
    end
end
EntropicCostEta = costDirectNoRounding - sum(P.*log(P),'all');

%% Compute Approximations for SVDs
%precompute SVDs
tic()
[U12,S12,V12] = svd(K12);
[U23,S23,V23] = svd(K23);
[U34,S34,V34] = svd(K34);
timeSVDsComput = toc();

% iteration over different ranks
costSVDs = zeros(size(rvec)); LogErrSVDs = zeros(size(rvec)); timeSVDs = zeros(numel(rvec),100);
for i = 2:numel(rvec)
    r = rvec(i)
    
    % Approximation based on SVDs
    K12approx = U12(:,1:r)*S12(1:r,1:r)*V12(:,1:r)';
    K23approx = U23(:,1:r)*S23(1:r,1:r)*V23(:,1:r)';
    K34approx = U34(:,1:r)*S34(1:r,1:r)*V34(:,1:r)';
    
    % Compute transport plan and cost
    for timeTest = 1:100
        tic();
        [x1,x2,x3,x4,~,iterSVDs] = MMSinkhorn4LineGraphLR(U12(:,1:r),S12(1:r,1:r)*V12(:,1:r)',U23(:,1:r),S23(1:r,1:r)*V23(:,1:r)',U34(:,1:r),S34(1:r,1:r)*V34(:,1:r)',m,m,m,m,1000,estop,1);
        [x1r,x2r,x3r,x4r,y1,y2,y3,y4] = Rounding4LineGraphLR(U12(:,1:r),S12(1:r,1:r)*V12(:,1:r)',U23(:,1:r),S23(1:r,1:r)*V23(:,1:r)',U34(:,1:r),S34(1:r,1:r)*V34(:,1:r)',x1,x2,x3,x4,m,m,m,m);
        timeSVDs(i,timeTest) = toc();
    end
    costSVDs(i) = costEvalLineGraphLR(U12(:,1:r),S12(1:r,1:r)*V12(:,1:r)',U23(:,1:r),S23(1:r,1:r)*V23(:,1:r)',U34(:,1:r),S34(1:r,1:r)*V34(:,1:r)',x1r,x2r,x3r,x4r,y1,y2,y3,y4,C12,C23,C34);
    costSVDsNoRounding(i) = costEvalLineGraphLR(U12(:,1:r),S12(1:r,1:r)*V12(:,1:r)',U23(:,1:r),S23(1:r,1:r)*V23(:,1:r)',U34(:,1:r),S34(1:r,1:r)*V34(:,1:r)',x1,x2,x3,x4,zeros(size(y1)),zeros(size(y2)),zeros(size(y3)),zeros(size(y4)),C12,C23,C34);
    itersSVDs(i) = iterSVDs;
    LogErr= 0;
    Entropy = 0;
    sumValP = 0;
    for a = 1:n
        for b = 1:n
            for c = 1:n
                for d = 1:n
                    val = K12(a,b)*K23(b,c)*K34(c,d);
                    valSVDs = K12approx(a,b)*K23approx(b,c)*K34approx(c,d);
                    LogErr = max(LogErr,abs(log(val)-log(valSVDs)));
                    valP = valSVDs * exp(x1(a)) * exp(x2(b)) * exp(x3(c)) * exp(x4(d));
                    sumValP = sumValP + valP;
                    valLogP = log(1/valP);
                    Entropy = Entropy + valP*valLogP;
                    Ptensor(a,b,c,d) = valP;
                end
            end
        end
    end
    sumValP
    LogErrSVDs(i) = LogErr;
    EntropicCostSVDs(i) = costSVDsNoRounding(i) + Entropy;
    
    % evaluate the theoretical error bound for the entropic cost function
    elog = LogErr; %only estimation available
    boundSVDs(i) = eta*(elog*(2+log(2/elog))+elog/2*log(n^4-1)+2*estop*log(1/estop*(n^4-1)))+(elog+2*estop)*Cnorm;
end
timeSVDs = mean(timeSVDs,2)

%% Compute Approximations using the TT-SVD
costTT = zeros(size(rvecTT)); LogErrTTsvd = zeros(size(rvecTT)); timeTT = zeros(numel(rvecTT),100);
K = @(I) K12(I(1),I(2)).*K23(I(2),I(3)).*K34(I(3),I(4));
TT=dmrg_cross(4,n,K,1e-12,'nswp',4,'rmin',30,'maxr',33);

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
        [x1,x2,x3,x4,~,iterTT] = MMSinkhorn4TT(G1,G2,G3,G4,m,m,m,m,1000,estop,1);
        [x1r,x2r,x3r,x4r,y1,y2,y3,y4] = Rounding4TT(G1,G2,G3,G4,x1,x2,x3,x4,m,m,m,m);
        timeTT(i,timeTest) = toc();
        %P(i,j,k,l) = (G1(1,i,:)'*G2(:,j,:)*G3(:,k,:)*G4(:,l,1)) *exp(x1(i)+x2(j)+x3(k)+x4(l)) + y1(i)*y2(j)*y3(k)*y4(l);
    end
    costTT(i) = costEvalTT(G1,G2,G3,G4,x1r,x2r,x3r,x4r,y1,y2,y3,y4,C12,C23,C34);
    costTTnoRounding(i) = costEvalTT(G1,G2,G3,G4,x1,x2,x3,x4,zeros(size(y1)),zeros(size(y2)),zeros(size(y3)),zeros(size(y4)),C12,C23,C34);
    itersTT(i) = iterTT;
    LogErr= 0;
    Entropy = 0;
    for a = 1:n
        for b = 1:n
            for c = 1:n
                for d = 1:n
                    val = K12(a,b)*K23(b,c)*K34(c,d);
                    valTTSVDs = squeeze(G1(1,a,:))'*squeeze(G2(:,b,:))*squeeze(G3(:,c,:))*squeeze(G4(:,d,1));
                    LogErr = max(LogErr,abs(log(val)-log(valTTSVDs)));
                    valP = valTTSVDs * exp(x1(a)) * exp(x2(b)) * exp(x3(c)) * exp(x4(d));
                    valLogP = log(1/valP);
                    Entropy = Entropy + valP*valLogP;
                end
            end
        end
    end
    LogErrTTsvd(i) = LogErr;
    EntropicCostTT(i) = costTTnoRounding(i) + Entropy;
    
    
    % evalutate the theoretical error bound for the entropic cost function
    elog = LogErr; %only estimation available
    boundTT(i) = eta*(elog*(2+log(2/elog))+elog/2*log(n^4-1)+2*estop*log(1/estop*(n^4-1)))+(elog+2*estop)*Cnorm;
end
timeTT = mean(timeTT,2);


%% maybe only do runtime for a bigger example
close all
set(gca,'fontsize',10)
set(figure(1), 'Position', [0 0 400 350])
%semilogy(rvecTT,abs(costTT-costDirect),'r'); hold on  %'$| \langle \mathcal{C} , \mathcal{P}_{\textsf{TT}} \rangle - \langle \mathcal{C} , \mathcal{P} \rangle  | $',...
%semilogy(rvecTT,abs(costSVDs(1,1:numel(rvecTT))-costDirect),'b'); %'$|\langle \mathcal{C} , \mathcal{P}_{\textsf{SVDs}} \rangle - \langle \mathcal{C} , \mathcal{P} \rangle  | $',...
semilogy(rvecTT,abs(EntropicCostSVDs(1:numel(rvecTT)) - EntropicCostEta ),'r-'); hold on
semilogy(rvecTT,abs(EntropicCostTT(1:numel(rvecTT)) - EntropicCostEta),'b-');
semilogy(rvecTT,LogErrTTsvd,'r:'); hold on
semilogy(rvecTT,LogErrSVDs(1:numel(rvecTT)),'b:');
semilogy(rvecTT,boundSVDs(1:numel(rvecTT)),'r--');
semilogy(rvecTT,boundTT(1:numel(rvecTT)),'b--');
leg = legend(...
    '$|V_{\mathcal{C}}^{1}(\mathcal{P}_{\textsf{SVDs}})-V_{\mathcal{C}}^{1}(\mathcal{P})|$',...
    '$|V_{\mathcal{C}}^{1}(\mathcal{P}_{\textsf{TT}})-V_{\mathcal{C}}^{1}(\mathcal{P})|$',...
    '$||\log(\mathcal{K}_{\textsf{TT}}) - \log(\mathcal{K}) ||_\infty$',...
    '$||\log(\mathcal{K}_{\textsf{SVDs}}) - \log(\mathcal{K}) ||_\infty$',...
    '$\varepsilon_{V_{\mathcal{C}}^1}$ with $\varepsilon_{\textsf{log}} = ||\log(\mathcal{K}_{\textsf{SVDs}}) - \log(\mathcal{K}) ||_\infty$',...
    '$\varepsilon_{V_{\mathcal{C}}^1}$ with $\varepsilon_{\textsf{log}} = ||\log(\mathcal{K}_{\textsf{TT}}) - \log(\mathcal{K}) ||_\infty$'...
    );
set(leg,'Interpreter','latex','Location','SouthWest');
xlabel('$r$','interpreter','latex')
xlim([2,rvecTT(end)])
print -depsc 'figures/SharpnessStudy'

