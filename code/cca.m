function [Wx, Wy, r] = cca(X,Y,K)

C = cov([X,Y]);
sx = size(X,2);
sy = size(Y,2);
Cxx = C(1:sx, 1:sx);
Cxy = C(1:sx, sx+1:sx+sy);
Cyx = Cxy';
Cyy = C(sx+1:sx+sy, sx+1:sx+sy) ;
invCyy = pinv(Cyy);
invCxx = pinv(Cxx);
% --- Calcualte Wx and r ---

[Wx,r] = eigs(invCxx*Cxy*invCyy*Cyx,K); % Basis in X




Wy = invCyy*Cyx*Wx;     % Basis in Y
Wy = Wy./repmat(sqrt(sum(abs(Wy).^2)),sy,1); % Normalize Wy
