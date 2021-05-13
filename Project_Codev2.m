%Nathan Lutes
%Project Simulation code
clear; clc; close all
global xti tau tauhist cnt evtcnt

%constants and initial conditions
states = 4; in = states; out = 2; L = 10;
cnt = 0;
evtcnt = [];
tauhist = [];

V = 2*rand([L*in,1])-1;
W = 2*rand([L*out,1])-1; 
x0 = [0.5; 0.5; 0; 0];
xhat0 = [0.5,0.5,0,0]';
xti = x0;

%simulation
dt = 0.01;
t0 = 0;
tf = 20;
t1 = t0:dt:tf;
x1 = [x0;xhat0;V;W];
%storage
x1hist = zeros(length(x1),length(t0:dt:tf));
for i = 1:length(t1)
    x1hist(:,i) = x1;
    xdot = ETNACv2(t1(i),x1);
    x1 = x1 + dt*xdot;
end

%calculate desired trajectory
w = 0.5;
xd = [sin(w*t1); cos(w*t1)];

%plot desired and actual trajectories
figure
plot(t1,x1hist(1,:))
hold on
plot(t1,x1hist(2,:))
plot(t1,xd(1,:),'--')
plot(t1,xd(2,:),'--')
hold off
legend('q1','q2','qd1','qd2')
xlabel('Time')
ylabel('Angle (Radians)')
title('Desired vs. Actual Trajectories')

%plot error 
figure
e = x1hist(1:2,:) - xd;
plot(t1,e(1,:));
hold on
plot(t1,e(2,:));
hold off
legend('e1','e2')
xlabel('Time')
ylabel('Angle')
title('Trajectory Error')

%plot tau history
figure
plot(t1,tauhist(1,:))
hold on
plot(t1,tauhist(2,:))
hold off
legend('tau1','tau2')
xlabel('Time')
ylabel('Torque')
title('Torque Histories')

%plot event count
figure
plot(t1,evtcnt)
xlabel('Time')
ylabel('Number of Events vs. Time')
title('Event count history')
