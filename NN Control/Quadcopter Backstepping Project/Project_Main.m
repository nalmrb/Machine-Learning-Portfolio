%Nathan Lutes
%Nonlinear Control Main Script
clear; clc; close all
addpath('S:\Documents\MATLAB\ODE_Solvers')
global e2old tOld e2dotold
%constants and paramters
in1 = 12; in2 = 9; out1 = 3; out2 = 3; L = 10; 
e2old = 0; tOld = 0; e2dotold = 0;

%define weights
V1 = 2*rand(in1*L,1)-1; V2 = 2*rand(in2*L,1)-1;
W1 = zeros(L*out1,1); W2 = zeros(L*out2,1);

%simulation
zeta0 = [0,0,0,0,0,0]'; eta0 = [0,0,0,0,0,0]'; intr1 = zeros(3,1); intr2 = zeros(3,1);
x0 = [zeta0; eta0; intr1; intr2; V1; V2; W1; W2];
t1 = 0:0.01:60;
x1 = ode4(@QuadcopterDyn,t1,x0);

%desired trajectory
xdhist = zeros(1,length(t1)); ydhist = zeros(1,length(t1)); zdhist = zeros(1,length(t1));
xd0 = 0; xdf = 20; tf = 60; dx = (xdf - xd0)/(tf^3);
yd0 = 0; ydf = 5; dy = (ydf - yd0)/(tf^3);
zd0 = 0; zdf = 10; dz = (zdf - zd0)/(tf^3);
DTM = [1 tf tf^2; 3 4*tf 5*tf^2; 6 12*tf 20*tf^2]; 
ax = DTM\[dx 0 0]'; ay = DTM\[dy 0 0]'; az = DTM\[dz 0 0]';
for i = 1:length(t1)
    xdhist(i) = ax'*[t1(i)^3 t1(i)^4 t1(i)^5]'; 
    ydhist(i) = ay'*[t1(i)^3 t1(i)^4 t1(i)^5]'; 
    zdhist(i) = az'*[t1(i)^3 t1(i)^4 t1(i)^5]';
end

%plots
Fsize = 15;
Pic_Width=7;
Pic_Height=7;

figure
plot(t1,xdhist,'LineWidth',3)
hold on
plot(t1,x1(:,1),'LineWidth',3)
plot(t1,ydhist,'LineWidth',3)
plot(t1,x1(:,2),'LineWidth',3)
plot(t1,zdhist,'LineWidth',3)
plot(t1,x1(:,3),'LineWidth',3)
hold off
legend('x_d','x','y_d','y','z_d','z')
xlabel('\bf{Time (s)}','FontWeight','bold','FontSize',20,'interpreter','latex')
ylabel('\bf{Position (m)}','FontWeight','bold','FontSize',20,'interpreter','latex')
ax = gca;ax.LineWidth = 3; ax.FontSize =Fsize; box on; ax.FontWeight='bold';
set(gcf,'PaperUnits','inches','PaperPosition',[0 0 Pic_Width Pic_Height]);
grid;set(gca,'MinorGridLineStyle','-');set(gca,'GridLineStyle','-.');box on;

%error
ex = x1(:,1)-xdhist'; ey = x1(:,2)-ydhist'; ez = x1(:,3)-zdhist';

figure
plot(t1,ex,'LineWidth',3)
hold on
plot(t1,ey,'LineWidth',3)
plot(t1,ez,'LineWidth',3)
hold off
legend('xError', 'yError', 'zError')
xlabel('\bf{Time (s)}','FontWeight','bold','FontSize',20,'interpreter','latex')
ylabel('\bf{X Error (m)}','FontWeight','bold','FontSize',20,'interpreter','latex')
ax = gca;ax.LineWidth = 3; ax.FontSize =Fsize; box on; ax.FontWeight='bold';
set(gcf,'PaperUnits','inches','PaperPosition',[0 0 Pic_Width Pic_Height]);
grid;set(gca,'MinorGridLineStyle','-');set(gca,'GridLineStyle','-.');box on;

%recalculate control
e2old = 0; tOld = 0; e2dotold = 0;
udhist = zeros(1,length(t1));
tauhist = zeros(3,length(t1));
for i = 1:length(t1)
    [ud,tau]=calculateControl(t1(i),x1(i,:)');
    udhist(i) = ud;
    tauhist(:,i) = tau;
end

figure
plot(t1,udhist,'LineWidth',3)
xlabel('\bf{Time (s)}','FontWeight','bold','FontSize',20,'interpreter','latex')
ylabel('\bf{Input Thrust (N)}','FontWeight','bold','FontSize',20,'interpreter','latex')
ax = gca;ax.LineWidth = 3; ax.FontSize =Fsize; box on; ax.FontWeight='bold';
set(gcf,'PaperUnits','inches','PaperPosition',[0 0 Pic_Width Pic_Height]);
grid;set(gca,'MinorGridLineStyle','-');set(gca,'GridLineStyle','-.');box on;

figure
plot(t1,tauhist(1,:),'LineWidth',3)
xlabel('\bf{Time (s)}','FontWeight','bold','FontSize',20,'interpreter','latex')
ylabel('\bf{Input Torque (N*m)}','FontWeight','bold','FontSize',20,'interpreter','latex')
ax = gca;ax.LineWidth = 3; ax.FontSize =Fsize; box on; ax.FontWeight='bold';
set(gcf,'PaperUnits','inches','PaperPosition',[0 0 Pic_Width Pic_Height]);
grid;set(gca,'MinorGridLineStyle','-');set(gca,'GridLineStyle','-.');box on;

figure
plot(t1,tauhist(2,:),'LineWidth',3)
xlabel('\bf{Time (s)}','FontWeight','bold','FontSize',20,'interpreter','latex')
ylabel('\bf{Input Torque (N*m)}','FontWeight','bold','FontSize',20,'interpreter','latex')
ax = gca;ax.LineWidth = 3; ax.FontSize =Fsize; box on; ax.FontWeight='bold';
set(gcf,'PaperUnits','inches','PaperPosition',[0 0 Pic_Width Pic_Height]);
grid;set(gca,'MinorGridLineStyle','-');set(gca,'GridLineStyle','-.');box on;

figure
plot(t1,tauhist(3,:),'LineWidth',3)
xlabel('\bf{Time (s)}','FontWeight','bold','FontSize',20,'interpreter','latex')
ylabel('\bf{Input Torque (N*m)}','FontWeight','bold','FontSize',20,'interpreter','latex')
ax = gca;ax.LineWidth = 3; ax.FontSize =Fsize; box on; ax.FontWeight='bold';
set(gcf,'PaperUnits','inches','PaperPosition',[0 0 Pic_Width Pic_Height]);
grid;set(gca,'MinorGridLineStyle','-');set(gca,'GridLineStyle','-.');box on;
