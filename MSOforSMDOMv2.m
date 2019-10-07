%Nathan Lutes
%MSO for DEQ parameter estimation
%9/26/2019

%This code applies a modified state observer to estimate a system online.
%The MSO is an observer containing a neural network with an online weight
%update rule

clear

%simulate system
%Generate Data
B = 8;    %define the correct answer
k = 25;
M = 1;
a2 = B/M;
a1 = k/M;
t = 2;
T = 0.001;
time = 0:T:t-T;
A = [0 1; -a1 -a2];
X = zeros(2,t);
X(:,1) = [5;0];
X_dot = zeros(2,t);
for i= 1:length(time)
    X_dot(:,i) = (A * X(:,i));
    X(:,i+1) = X(:,i) + T*(A*X(:,i));   %rectangular integration
end

%MSO
%initial model parameters
Ae = [0 1; 0.75*A(2,1) 0.75*A(2,2)];
aehist = zeros(2,length(time));
r = 0.3;   %learning rate
K = 20*eye(2);
Q = eye(2);
P = 0.5*Q/K;   %solution to Lyapunov stability equation
T = 0.001;
Xhat = zeros(size(X));
Xhat(:,1) = [5;0];   %initial Xhat equal to initial X
W = zeros(2, 2); %initialize weights
phi = [Xhat(1,1) Xhat(2,1)]'; %initialize phi
e = X(:,1)-Xhat(:,1);   %initial error

%MSO Loop
for i = 2:length(time)+1
    %calculate current value for MSO
    Xhat(:,i) = Xhat(:,i-1) + T*(Ae * Xhat(:,i-1) + W' * phi + K*e);
    %calculate estimated parameters for previous iteration
    aehist(:,i) = [Ae(2,1); Ae(2,2)] + [W(1,2);W(2,2)] + K(2,2)*e(2);
    %update weights on current error using previous phi
    W = W + T*(r*phi*e'*P);
    %calculate error for current iteration
    e = X(:,i) - Xhat(:,i);
    %calculate phi for current iteration
    phi = [Xhat(1,i) Xhat(2,i)]';
end

%plot Xhat vs X
figure()
hold on
plot(time,X(1,1:end-1),'b')
plot(time,Xhat(1,1:end-1),'r')
hold off
title('MSO position vs Actual System')
xlabel('Time')
ylabel('System Response Position')
legend('Actual','MSO')

figure()
hold on
plot(time,X(2,1:end-1),'b')
plot(time,Xhat(2,1:end-1),'r')
hold off
title('MSO velocity vs Actual System')
xlabel('Time')
ylabel('System Response Velocity')
legend('Actual','MSO')

%plot actual system parameters vs estimated system parameters
figure()
hold on
plot(time(2:end),A(2,1),'b')
plot(time(2:end),aehist(1,2:end-1), 'r')
hold off
title('actual system parameters vs estimated system parameters MSO')
xlabel('time')
ylabel('a1')
figure()
hold on
plot(time(2:end),A(2,2),'b')
plot(time(2:end),aehist(2,2:end-1), 'r')
hold off
title('actual system parameters vs estimated system parameters MSO')
xlabel('time')
ylabel('a2')