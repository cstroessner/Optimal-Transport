% rounding of a tensor A of order 3 to satisfy marginals r1 r2 r3

function B = Rounding3(A,r1,r2,r3)
rA1 = squeeze(sum(A,[2,3]));
v = min(r1./rA1,ones(size(r1)));
for i = 1:numel(r1)
    A(i,:,:) = A(i,:,:).*v(i);
end
rA2 = squeeze(sum(A,[1,3]))';
v = min(r2./rA2,ones(size(r2)));
for i = 1:numel(r2)
    A(:,i,:) = A(:,i,:).*v(i);
end
rA3 = squeeze(sum(A,[1,2]));
v = min(r3./rA3,ones(size(r3)));
for i = 1:numel(r3)
    A(:,:,i) = A(:,:,i).*v(i);
end
rA1 = squeeze(sum(A,[2,3]));
rA2 = squeeze(sum(A,[1,3]))';
rA3 = squeeze(sum(A,[1,2]));

e1 = r1-rA1;
e2 = r2-rA2;
e3 = r3-rA3;
B = zeros(size(A));
for I = 1:numel(A)
    [a,b,c] = ind2sub(size(A),I);
    B(a,b,c) = e1(a)*e2(b)*e3(c);
end

B = A + 1/(norm(e1,1)^2) * B;
end