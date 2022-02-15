% rounding of a tensor A of order 4 to satisfy marginals r1 r2 r3 r4

function B = Rounding4(A,r1,r2,r3,r4)
rA1 = squeeze(sum(A,[2,3,4]));
v = min(r1./rA1,ones(size(r1)));
for i = 1:numel(r1)
    A(i,:,:,:) = A(i,:,:,:).*v(i);
end
rA2 = squeeze(sum(A,[1,3,4]))';
v = min(r2./rA2,ones(size(r2)));
for i = 1:numel(r2)
    A(:,i,:,:) = A(:,i,:,:).*v(i);
end
rA3 = squeeze(sum(A,[1,2,4]));
v = min(r3./rA3,ones(size(r3)));
for i = 1:numel(r3)
    A(:,:,i,:) = A(:,:,i,:).*v(i);
end
rA4 = squeeze(sum(A,[1,2,3]));
v = min(r4./rA4,ones(size(r4)));
for i = 1:numel(r4)
    A(:,:,:,i) = A(:,:,:,i).*v(i);
end

rA1 = squeeze(sum(A,[2,3]));
rA2 = squeeze(sum(A,[1,3]))';
rA3 = squeeze(sum(A,[1,2]));
rA4 = squeeze(sum(A,[1,2,3]));

e1 = r1-rA1;
e2 = r2-rA2;
e3 = r3-rA3;
e4 = r4-rA4;

B = zeros(size(A));
for I = 1:numel(A)
    [a,b,c,d] = ind2sub(size(A),I);
    B(a,b,c,d) = e1(a)*e2(b)*e3(c)*e4(d);
end

B = A + 1/(norm(e1,1)^3) * B;
end