function [qdot] = ETNACv2(t,q)
%This function implements an event triggered neural adaptive controller for
%use in NN project

%define event triggered state
global xti tau tauhist cnt evtcnt

%constants
states = 4; est = states;
in = states; L = 10; out = 2; Lphi = 1; Lf = 1;
K2 = (0.5)*[40*eye(2),0*eye(2);zeros(2), 20*eye(2)]; 
Vsize = L*in; Wsize = L*out; sigma = 1; Q = eye(states);
P=0.0031*Q;
beta = sigma/(1+(norm(P)*(norm(Lphi) + norm(Lf)))^2);

% beta = sigma/(1+(norm(P)*(norm(Lphi) + norm(Lf)))^2);
a1 = 1; a2 = 1; m1 = 1; m2 = 2.3; g = 9.8;
gam = .4; %gam = 30;
Fv=0.05*eye(2); dist=0.05*[0.01;0.01];
alpha1 = .01; %case 1
alpha2 = 0.005; %case 2
alpha3 = 0.1; %case 3
alpha = alpha1;

%indices
indexInput = 1:states;
indexEst = indexInput(end) + 1:indexInput(end) + est;
indexV = indexEst(end) + 1:indexEst(end) + Vsize;
indexW = indexV(end) + 1:indexV(end) + Wsize;

%get input measurements
x = q(indexInput); xhat = q(indexEst); V = reshape(q(indexV),[in,L]);
W = reshape(q(indexW),[L,out]);

%calculate desired trajectory
w = 0.5;
xd = [sin(w*t); cos(w*t)];
xddot = [w*cos(w*t); -w*sin(w*t)];
F = [0 0 1 0; 0 0 0 1; -w^2 0 0 0; 0 -w^2 0 0]*[xd; xddot];

%Robot system
M = [(m1+m2)*a1^2 + m2*a2^2 + 2*m2*a1*a2*cos(x(2))...
    m2*a2^2 + m2*a1*a2*cos(x(2));...
    m2*a2^2 + m2*a1*a2*cos(x(2)) m2*a2^2];
N = [-m2*a1*a2*(2*x(3)*x(4) + x(4)^2)*sin(x(2)) + (m1+m2)*g*a1*cos(x(1)) + m2*g*a2*cos(x(1) + x(2));...
    m2*a1*a2*(x(3)^2)*sin(x(2)) + m2*g*a2*cos(x(1) + x(2))];
Mxti = [(m1+m2)*a1^2 + m2*a2^2 + 2*m2*a1*a2*cos(xti(2))...
    m2*a2^2 + m2*a1*a2*cos(xti(2));...
    m2*a2^2 + m2*a1*a2*cos(xti(2)) m2*a2^2];
Nxti = [-m2*a1*a2*(2*xti(3)*xti(4) + xti(4)^2)*sin(xti(2)) + (m1+m2)*g*a1*cos(xti(1)) + m2*g*a2*cos(xti(1) + xti(2));...
    m2*a1*a2*(xti(3)^2)*sin(xti(2)) + m2*g*a2*cos(xti(1) + xti(2))];
fxti = [xti(3:4);-Mxti\Nxti];
B = [zeros(2);inv(M)];
Bs = [-inv(M);inv(M)];
Us = [1;1];
Baug = [B,Bs];

%calculate neural network
phi = 1./(1 + exp(-(V'*x)));
dhat = W'*phi;
d = Fv*xd+dist;

%event triggered
%decide if event is triggered and xti should be updated
beta=3000/(1+(2*norm(P))^2);
er = x - [xd; xddot];
eevt = x - xti;
if norm(eevt(1:2))^2 <= alpha*beta*norm(er(1:2))^2
    xti = x;
    %calculate control input
    K = 0.5*[30*eye(2), 0*eye(2);0*eye(2),15*eye(2)];
    tau = Baug\(F- K*(xhat-[xd;xddot]) - K2*(xti-xhat) - B*dhat - [x(3); x(4); -M\N] + Bs*Us);
    cnt = cnt + 1;
    %update weights
    eDC = xti - [xd;xddot];
    Wdot = gam*(phi*eDC'*P)*B;
else
    Wdot = zeros(size(W));
end

%periodic
% xti = x;
%calculate control input
% K = 0.5*[30*eye(2), 0*eye(2);0*eye(2),15*eye(2)];
% tau = Baug\(F- K*(xhat-[xd;xddot]) - K2*(xti-xhat) - B*dhat - [x(3); x(4); -M\N] + Bs*Us);
% %update weights
% eDC = xti - [xd;xddot];
% Wdot = gam*(phi*eDC'*P)*B;

%record control input
tauhist = [tauhist tau];
%record event history
evtcnt = [evtcnt cnt];

%calculate state and weight update laws
xdot = [x(3); x(4); -M\N] + Baug*tau - Bs*Us + B*d;
xhatdot = fxti + Baug*tau - Bs*Us +B*dhat + K2*(xti-xhat);
Vdot = zeros(size(V));

qdot = [xdot; xhatdot; reshape(Vdot,[L*in,1]); reshape(Wdot,[L*out,1])];

end

