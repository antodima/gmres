function [v, s] = householder_vector(x)
s = -sgn(x(1)) * norm(x);
v = x;
v(1) = v(1) - s;
v = v / norm(v);