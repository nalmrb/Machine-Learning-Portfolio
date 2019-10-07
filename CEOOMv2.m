%Nathan Lutes
%9/27/2019

%%% This program employs the CE method for online estimation
clc 
clear
ask = input('close figures? (1/0): ');
if ask == 1
    try
        close ALL HIDDEN
    catch
        disp('no figures to close\n')
    end
end

%Create Storage
t = 2;
T = 0.001;
time = 0:T:t;
X = zeros(2,length(time));
Xdot = zeros(2,length(time));
Xhat = zeros(2,length(time));
Xhatdot = zeros(2,length(time)); 

%Actual System parameters
B = 8;    
k = 25;
M = 1;
a2 = B/M;
a1 = k/M;
A = [0 1; -a1 -a2];

%initial estimate parameters
Ae = [0 1; 0.75*A(2,1) 0.75*A(2,2)];
mui = [5 * Ae(2,1), 5 * Ae(2,2)];
mu = mui;
N = 10000;
Ne = N*0.01;
alpha = 0.9;    %smoothing function parameter
stopC = 1e-6;   %stop condition for max variance
Schecki = 10000;
Scheck = Schecki;
d = 0;
dd = 0;
n = 10;     %number of samples to run CE algorithm
j = -1;
ae = zeros(1,N);
S = zeros(1,N);

%initial conditions
X(:,1) = [5;0];
Xhat(:,1) = [5;0];

%simulation
for i= 1:length(time)
    j = j + 1; %count number of iterations
    %simulate actual system
    Xdot(:,i) = (A * X(:,i)) + normrnd(0,1,2,1);
    X(:,i+1) = X(:,i) + T*(A*X(:,i)) + normrnd(0,1,2,1);   
    %simulate estimated system
    Xhatdot(:,i) = (Ae * X(:,i));
    Xhat(:,i+1) = Xhat(:,i) + T*(Ae*Xhat(:,i));
    %calculate error
    e = X(:,i) - Xhat(:,i);
    if j >= n && max(abs(e)) > 1e-3
        while dd < 10
            %set mu
            if dd>0
                mu = [1.5 * Ae(2,1), 1.5 * Ae(2,2)];
            end
            %set variance
            var = [1000 * max(abs(e)), 1000 * max(abs(e))];
            %reset Scheck
            Scheck = Schecki;
            %run CE algorithm until convergence
            while max(var) > stopC && d < 5
                for k = 1:N
                    %Collect Samples
                    ae(1,k) = normrnd(mu(1),var(1));
                    ae(2,k) = normrnd(mu(2),var(2));
                    %Evaluate function
                    S(k) = (1/N) * sum((Xdot(2,i-j:i) - (- ae(1,k) * X(1,i-j:i) - ae(2,k) * X(2,i-j:i))).^2);
                end
                Samps = sortrows([ae; S]',3);
                index = find(Samps(:,3) <= Scheck,1);
                if index > Ne
                    Elite = Samps(1:Ne,:);
                else
                    Elite = Samps(1:index,:);
                    Ne = index;
                end
                %update parameters
                if ~isempty(Elite)    %check if none of the samples were smaller than criteria
                    %compute mean and variance tilde
                    muT = (1/Ne) * sum(Elite(:,1:2),1);
                    varT = (1/Ne) .* sum(((Elite(:,1:2) - repmat(muT,Ne,1)).^2),1);
                    %smoothing
                    mu = alpha*muT + (1 - alpha)*mu;
                    var = alpha*varT + (1 - alpha)*var;
                    %set Scheck for next iteration
                    Scheck = mean(Elite(:,3));
                    d = 0;  %reset d
                    Ne = 0.01*N;   %reset number of elite samples
                else
                    d = d + 1;   %Scheck should be same as last iteration
                end
            end
            if d == 5 %algorithm did not converge
                d = 0;
                fprintf('did not converge\n')
                dd = dd + 1;
                continue
            else  %algorithm did converge
                fprintf('converged\n')
                %reset parameters
                d = 0;
                %update parameters of system using found parameters
                index = find(Elite(:,3) == min(Elite(:,3)));   %finds index of minimum value in S
                %update Ae matrix
                Ae = [0 1;-Elite(index,1) -Elite(index,2)];
                %if the algorithm updates, update Xhat as well
                Xhat(:,i+1) = X(:,i+1);
                Xhatdot(:,i+1) = Xhat(:,i+1);
                break
            end
        end
        dd = 0;
        j = 0;  %reset j
    else
        continue
    end
end

%plot system response and estimate response
%plot position
figure()
hold on
plot(time,Xhat(1,1:end-1), 'b');
plot(time, X(1,1:end-1), 'r');
hold off
title('System with estimated parameters vs. Actual System')
xlabel('time')
ylabel('System Response Position')
legend('estimated', 'actual')
%plot velocity
figure()
hold on
plot(time,Xhat(2,1:end-1), 'b');
plot(time, X(2,1:end-1), 'r');
hold off
title('System with estimated parameters vs. Actual System')
xlabel('time')
ylabel('System Response Velocity')
legend('estimated', 'actual')