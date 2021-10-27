% GMRES for large and sparse linear system

m = 1000;
A = 10*speye(m) + sprandn(m, m, 0.01);
b = randn(size(A,1), 1);
[Q, H] = arnoldi(A, b, 50);
r = nan(50,1);
% tests several ’slices’ of Q,H
% this simulates Arnoldi with various values of n.
for n = 1:50
    y = H(1:n+1, 1:n) \ eye(n+1, 1) * norm(b);
    x = Q(:, 1:n) * y;
    r(n) = norm(A*x - b);
end
semilogy(r)




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [v, s] = householder_vector(x)
    s = -sgn(x(1)) * norm(x);
    v = x;
    v(1) = v(1) - s;
    v = v / norm(v);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Q, A] = myqr(A)
    [m, n] = size(A);
    Q = eye(m);
    for j = 1:n-1
        [v, s] = householder_vector(A(j:end, j));
        A(j,j) = s; A(j+1:end,j) = 0;
        A(j:end,j+1:end) = A(j:end,j+1:end) - 2*v*(v'*A(j:end,j+1:end));
        Q(:, j:end) = Q(:, j:end) - Q(:,j:end)*v*2*v';
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Q, H] = arnoldi(A, b, n)
    m = length(A);
    Q = zeros(length(b), n);
    Q(:,1) = b / norm(b);
    H = zeros(min(n+1,n),m);

    for k=1:n
        z = A * Q(:,k);
        for i=1:k
            H(i,k) = Q(:,i)' * z;
            z = z - H(i,k) * Q(:,i);
        end
        if k < m
           H(k+1,k) = norm(z);
           if H(k+1,k) == 0, return, end
           Q(:,k+1) = z / H(k+1,k);
        end
    end
end