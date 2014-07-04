function [hx, W] = mDAsd(xx, idx)
% xx : dxn input
% idx: indices of 'pivot' features

% hx: dxn hidden representation
% W: dx(d+1) mapping

lambda = 1e-05;
[d, n] = size(xx);

xfreq = xx(idx,:);
normvec = sum(xx,2);
normvec = normvec + lambda*ones(d,1);
% scatter matrix S (P)
P = xfreq*xx';
Q = sparse(d,d);
Q(1:d+1:end) = normvec;
W = P/Q;
hx = W*xx;
hx = tanh(hx);
