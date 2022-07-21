% the following code demonstrates how the elementwise exponential increases the rank as stated in the introduction

clear
close all
rng(1)

n = 1000;

for i=1:100
    A = rand(n,5)*rand(5,n);
    r(i) = sum(svd(exp(A)) > 1e-10);
end
r = mean(r)
