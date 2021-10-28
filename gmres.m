% GMRES for large and sparse linear system

m = 1000;
n = 50;
A = 10*speye(m) + sprandn(m, m, 0.01);
b = randn(size(A,1), 1);
[Q, H] = arnoldi(A, b, n);
r = nan(n,1);
% tests several ’slices’ of Q,H
% this simulates Arnoldi with various values of n.
for i = 1:n
    % min|| Qn+1*Hn*y - ||b||*e1 ||
    y = H(1:i+1, 1:i) \ eye(i+1, 1) * norm(b);
    x = Q(:, 1:i) * y;
    r(i) = norm(A*x - b);
end
figure;
semilogy(r);
hold on;
semilogy(r+0.01);
legend('r1','r2','Location','southwest');
hold off;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Q, H] = arnoldi(A, b, n) 
    m = length(A); % rows of A
    Q = zeros(length(b), n); % orthonormal matrix
    Q(:,1) = b / norm(b);
    H = zeros(n+1, n); % Hessenberg matrix

    for k = 1:n % cost: n + O(mn^2)
        w = A * Q(:,k); % cost: n "matvec" products
        for i = 1:k
            H(i,k) = Q(:,i)' * w; % O(mn)
            w = w - H(i,k) * Q(:,i); % O(n)
        end
       H(k+1,k) = norm(w);
       %if H(k+1,k) == 0, return, end  % lucky breakdown
       Q(:,k+1) = w / H(k+1,k);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [v, s] = householder_vector(x)
% Computes Householder vector of vector x.
%
% Parameters
% ----------
%   x: vector
%
% Returns
% -------
%   v: householder vector
%   s: norm of x

    s = -sgn(x(1)) * norm(x);
    v = x;
    v(1) = v(1) - s; 
    v = v / norm(v);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Q, R] = myqr(A)
% Q: orthogonal
    [m, n] = size(A); 
    Q = eye(m);
    for j = 1:n-1
        [v, s] = householder_vector(A(j:end, j));
        A(j,j) = s; A(j+1:end,j) = 0;
        A(j:end,j+1:end) = A(j:end,j+1:end) - 2*v*(v'*A(j:end,j+1:end));
        Q(:, j:end) = Q(:, j:end) - Q(:,j:end)*v*2*v';
    end
    R = A;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Q, R] = qr_hess(H)
% Computes QR factorization of the Hessenberg matrix.
%   - http://algorithm-interest-group.me/assets/slides/Eigenvalue_algorithms.pdf
%   - http://www.math.usm.edu/lambers/mat610/sum10/lecture9.pdf
%
%   - https://www3.math.tu-berlin.de/Vorlesungen/SoSe11/NumMath2/Materials/qr__iteration_eng.pdf
%   - https://github.com/lucasbekker/GMRES/blob/master/GMRES.m
%
% Parameters
% ----------
%   H: Hessenberg matrix
%
% Returns
% -------
%   Q: orthogonal matrix
%   H: upper triangular matrix

    R = H(1:end-1,:); % remove last row
%     [m, n] = size(R);
%     for j = 1:n-1
%         R(j+1:end,j) = 0;
%     end
    
    
end





