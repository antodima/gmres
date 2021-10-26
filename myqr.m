function [Q, A] = myqr(A)
[m, n] = size(A);
Q = eye(m);
for j = 1:n-1
[v, s] = householder_vector(A(j:end, j));
A(j,j) = s; A(j+1:end,j) = 0;
A(j:end,j+1:end) = A(j:end,j+1:end) - ...
2*v*(v’*A(j:end,j+1:end));
Q(:, j:end) = Q(:, j:end) - Q(:,j:end)*v*2*v’;
end