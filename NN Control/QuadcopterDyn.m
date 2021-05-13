function [xdot] = QuadcopterDyn(t,x)
%Quadcopter dynamics for nonlinear control final
global e2old tOld e2dotold
%constants
g = 9.81; L = 10; in1 = 12; in2 = 9; out1 = 3; out2 = 3; m = 1;
Lamb1 = diag(1*ones(1,3)); Lamb2 = diag(1*ones(1,3)); Kr1 = diag(15*ones(1,3));
Kr2 = diag(15*ones(1,3)); Ki1 = diag(5*ones(1,3)); Ki2 = diag(5*ones(1,3));
V1index = 19+L*in1-1; V2index = V1index + L*in2;
W1index = V2index + L*out1;
W2index = W1index + L*out2;
F1 = diag(.02*ones(1,10)); F2 = F1; k = 0.01;

%divide x into states
zeta = x(1:6); eta = x(7:12); psi = eta(1); th = eta(2); phi = eta(3);
intr1 = x(13:15); intr2 = x(16:18); V1 = reshape(x(19:V1index),in1,L);
V2 = reshape(x(V1index+1:V2index),in2,L);
W1 = reshape(x(V2index+1:W1index),L,out1);
W2 = reshape(x(W1index+1:W2index),L,out2);

% system dynamics
Teta = [-sin(th) 0 1; cos(th)*sin(psi) cos(psi) 0; cos(th)*cos(psi)...
    -sin(psi) 0];
Tetadot = [-cos(th)*eta(5) 0 0; (-sin(th)*eta(5)*sin(psi) + cos(psi)*eta(4)*cos(th))...
    -sin(psi)*eta(4) 0; (-sin(th)*eta(5)*cos(psi) - sin(psi)*eta(4)*cos(th))...
    -cos(psi)*eta(4) 0];
Sigma = diag([5,5,5]); J = Teta'*Sigma*Teta;
Axyz = [zeta(4)^2 * eta(6); zeta(5)^2 * eta(5); zeta(6)^2 * eta(4)];
Aeta = [zeta(4)*eta(6)^2; zeta(5)*eta(5)^2; zeta(6)*eta(4)^2];
c11 = (2*Sigma(2,2)*cos(th)^(2)*sin(psi)*cos(psi) - 2*Sigma(3,3)*cos(th)^(2)...
    *cos(psi)*sin(psi))*eta(4) + ((Sigma(2,2)-Sigma(3,3))*cos(th)*...
    (cos(psi)^(2)-sin(psi)^(2)))*eta(5);
c12 = (2*Sigma(1,1)*sin(th)*cos(th) - 2*Sigma(2,2)*sin(psi)^(2)*sin(th)*cos(th)...
    - 2*Sigma(3,3)*sin(th)*cos(th)*cos(psi)^(2))*eta(4) - ((Sigma(2,2)-Sigma(3,3))...
    *sin(psi)*cos(psi)*sin(th))*eta(5) - Sigma(1,1)*cos(th)*eta(6);
c13 = 0;
c21 = ((Sigma(2,2)-Sigma(3,3))*cos(th)*(cos(psi)^(2)-sin(psi)^(2)))*eta(4)...
    + (-2*Sigma(2,2)*cos(psi)*sin(psi) + 2*Sigma(3,3)*sin(psi)*cos(psi))*eta(5);
c22 = (-(Sigma(2,2)-Sigma(3,3))*sin(th)*sin(psi)*cos(psi))*eta(4);
c23 = 0; c31 = 0; c32 = -Sigma(1,1)*cos(th)*eta(4); c33 = 0;
c = [c11 c12 c13; c21 c22 c23; c31 c32 c33];
C = 2*Teta'*Sigma*Tetadot - 0.5*c;

%desired trajectory
xd0 = 0; xdf = 20; tf = 60; dx = (xdf - xd0)/(tf^3);
yd0 = 0; ydf = 5; dy = (ydf - yd0)/(tf^3);
zd0 = 0; zdf = 10; dz = (zdf - zd0)/(tf^3);
DTM = [1 tf tf^2; 3 4*tf 5*tf^2; 6 12*tf 20*tf^2];
ax = DTM\[dx 0 0]'; ay = DTM\[dy 0 0]'; az = DTM\[dz 0 0]';
xd = ax'*[t^3 t^4 t^5]'; yd = ay'*[t^3 t^4 t^5]'; zd = az'*[t^3 t^4 t^5]';
xddot = ax'*[3*t^2 4*t^3 5*t^4]'; yddot = ay'*[3*t^2 4*t^3 5*t^4]';
zddot = az'*[3*t^2 4*t^3 5*t^4]';
zetad = [xd; yd; zd; xddot; yddot; zddot];
xdddot = ax'*[6*t 12*t^2 20*t^3]'; ydddot = ay'*[6*t 12*t^2 20*t^3]';
zdddot = az'*[6*t 12*t^2 20*t^3]';
zetadddot = [xdddot; ydddot; zdddot];
psid = 0;

%control
e1 = zetad(1:3) - zeta(1:3); e1dot = zetad(4:6) - zeta(4:6);
r1 = e1dot + Lamb1*e1; X1 = [zeta(1:3); r1; eta(1:6)];
mu1 = (1./(1+exp(-V1'*X1)));
y1 = W1'*mu1;
Fhatd = m*zetadddot - m*Lamb1^(2)*e1 - [0; 0; -m*g] - y1 + Kr1*r1 +...
    Ki1*intr1;
a = -Fhatd(1); b = (Fhatd(2)^(2)+Fhatd(3)^(2))^(1/2);
ud = (a^2 + b^2)^(1/2); thd = atan(a/b); phid = atan(Fhatd(2)/Fhatd(3));
e2 = [psid-psi; thd-th; phid-phi];
if (t-tOld) ~= 0
    e2dot = (e2 - e2old)/(t - tOld);
    r2 = e2dot + Lamb2*e2;
else
    e2dot = e2dotold;
    r2 = e2dot + Lamb2*e2;
end
X2 = [eta(1:3); r1; r2];
mu2 = (1./(1+exp(-V2'*X2)));
y2 = W2'*mu2;
tau = y2 + Kr2*r2 + Ki2*intr2;
e2old = e2; tOld = t; e2dotold = e2dot; 
r = [r1;r2];

%derivative calculation
zetadot = [zeta(4); zeta(5); zeta(6); (-ud/m)*sin(th) + (1/m)*Axyz(1);...
    (ud/m)*cos(th)*sin(phi) + (1/m)*Axyz(2); (ud/m)*cos(th)*cos(phi) - g + (1/m)*Axyz(3)];
etadot = [eta(4); eta(5); eta(6); J\(tau - C*[eta(4); eta(5); eta(6)] + Aeta)];
V1dot = zeros(L*in1,1); V2dot = zeros(L*in2,1);
W1dot = F1*mu1*r1'-k*norm(r)*F1*W1; W2dot = F2*mu2*r2'-k*norm(r)*F2*W2;
xdot = [zetadot; etadot; r1; r2; V1dot; V2dot;reshape(W1dot,L*out1,1);reshape(W2dot,L*out2,1)];
end