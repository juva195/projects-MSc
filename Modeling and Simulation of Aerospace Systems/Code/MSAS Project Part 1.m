%% SETTINGS

% Defaults "grid on"
set(groot,'defaultAxesXGrid','on','defaultAxesXGridMode','manual');
set(groot,'defaultAxesYGrid','on','defaultAxesYGridMode','manual');
set(groot,'defaultAxesZGrid','on','defaultAxesZGridMode','manual');

% Set grid transparency
set(groot,'defaultAxesGridAlphaMode','manual','defaultAxesGridAlpha',0.25);

% Defaults "grid minor"
% set(groot,'defaultAxesXMinorGrid','on','defaultAxesXMinorGridMode','manual');
% set(groot,'defaultAxesYMinorGrid','on','defaultAxesYMinorGridMode','manual');
% set(groot,'defaultAxesZMinorGrid','on','defaultAxesZMinorGridMode','manual');

% Set grid minor transparency
% set(groot,'defaultAxesMinorGridAlphaMode','manual','defaultAxesMinorGridAlpha',0.25);

% Interpreter
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultColorbarTickLabelInterpreter', 'latex');

% Set font, font style and size
set(groot, 'defaultAxesFontWeight', 'bold');
set(groot, 'defaultAxesFontName', 'Palatino Linotype', 'defaultTextFontName', 'Palatino Linotype');
set(groot, 'defaultAxesFontSize',12);
set(groot, 'defaultLegendFontSize', 12,'DefaultLegendFontSizeMode','manual');

% Set Figure Position
% set(groot, 'defaultFigurePosition', [470, 360, 700, 430])

% Set Figure Colormap
set(groot, 'defaultFigureColormap', turbo(256));
set(groot, 'defaultSurfaceEdgeAlpha', 0.1);

%Set LineWidth
set(groot, 'defaultLineLineWidth', 1);

% Set Figure background colors
set(groot, 'defaultFigureColor', 'w'); % Border color, outside plotted part
set(groot, 'defaultAxesColor', 'w'); % Plotted part backgroud color

%% EXERCISE 1

clc
clear
close all

% Define initial variables
a = [10, 13, 8, 10];                    % Lenght of the rods [cm]
toll = 1e-5;                            % Tollerances
alpha0 = [-0.1, 2/3*pi];                % Initial alpha values
nmax = 100;                             % Maximum iteration number

% Computing of the analytical derivative
syms alpha beta
fsym = a(1)/a(2)*cos(beta) - a(1)/a(4)*cos(alpha) - cos(beta-alpha) + (a(1)^2 + a(2)^2 - a(3)^2 + a(4)^2)/(2*a(2)*a(4));
dfsym = diff(fsym, alpha);

f = matlabFunction(fsym);
df = matlabFunction(dfsym);

% Solve the problem through NS and fzero and plot the error between the two
beta = linspace(0,2/3*pi,500);          % Definition of beta

% Initialize the alpha and sol vectors
n = max(size(beta));
m = max(size(alpha0));

alpha = zeros(n,m);
it = zeros(n,m);
real = zeros(n,m);

% Compute and the solution for each alpha0
for j = 1:m
    for i = 1:n
        fun = @(alpha) f(alpha,beta(i));
        dfun = @(alpha) df(alpha,beta(i));
        real(i,j) = fzero(fun,alpha0(j));
        [alpha(i,j), it(i,j)] = NS(alpha0(j),nmax,toll,fun,dfun);
    end
end

% Plot the solutions
figure
plot(beta,alpha(:,1))
hold on
plot(beta,alpha(:,2))
xlim([0 2.0944])
xlabel('$\beta$ [rad]');
ylabel('$\alpha$ [rad]');
title('NS solution of the analytical derivative')
legend('$\alpha_0 = -0.1$', '$\alpha_0 = \frac{2}{3} \pi$')
legend('Location','northwest')
hold off

ic1 = max(abs(alpha(:,1)-real(:,1))); 
ic2 = max(abs(alpha(:,2)-real(:,2))); 
categories = {'$\alpha_0 = -0.1$', '$\alpha_0 = \frac{2}{3} \pi$'}; 
figure;
b = bar([ic1, ic2], 'FaceColor', 'flat'); 
b.CData(1,:) = [0 0.4470 0.7410]; 
b.CData(2,:) =  [0.8500 0.3250 0.0980]; 
ylim([0 3e-10])
xticks(1:2); 
xticklabels(categories);
title('Accuracy Comparison');
xlabel('Initial Conditions');
ylabel('Maximum Error');
grid on;


% Compute the derivative through finite differences
h = sqrt(eps);
alphaFD = zeros(n,m);
itFD = zeros(n,m);

for j = 1:m
    for i = 1:n
        fun = @(alpha) f(alpha,beta(i));

        [alphaFD(i,j), itFD(i,j)] = NS_FD(alpha0(j),nmax,toll,fun,h);
    end
end

% Plot the solutions
figure
plot(beta,alphaFD(:,1))
hold on
plot(beta,alphaFD(:,2))
plot(beta,alpha(:,1),"LineStyle","--","Color",[135/255, 206/255, 235/255])
plot(beta,alpha(:,2),"LineStyle","--","Color",[255/255, 200/255, 124/255])
xlim([0 2.0944])
xlabel('$\beta$ [rad]');
ylabel('$\alpha$ [rad]');
title('NS solution of the FD derivative')
legend('FD: $\alpha_0 = -0.1$', 'FD: $\alpha_0 = \frac{2}{3} \pi$','Analyitcal: $\alpha_0 = -0.1$', 'Analyitcal: $\alpha_0 = \frac{2}{3} \pi$')
legend('Location','northwest')
hold off

ic1 = max(abs(alpha(:,1) - real(:,1))); 
ic2 = max(abs(alpha(:,2) - real(:,2))); 
ic3 = max(abs(alphaFD(:,1) - real(:,1))); 
ic4 = max(abs(alphaFD(:,2) - real(:,2))); 
data = [ic1, ic2, ic3, ic4];
figure;
b = bar([1,2,4,5],data, 'FaceColor', 'flat'); 
b.CData(1,:) = [0 0.4470 0.7410];  
b.CData(2,:) = [0.8500 0.3250 0.0980];  
b.CData(3,:) = [0 0.4470 0.7410]; 
b.CData(4,:) = [0.8500 0.3250 0.0980];
ylim([0 3e-10])
xticks([1.5 4.5]); % Centered ticks for groups
xticklabels({'Analytical', 'Finite Differences'}); 
title('Accuracy Comparison');
xlabel('Methods');
ylabel('Maximum Error');
grid on;
hold on; 
h_blue = bar(nan, 'FaceColor', [0 0.4470 0.7410]); 
h_orange = bar(nan, 'FaceColor', [0.8500 0.3250 0.0980]); 
legend([h_blue, h_orange], {'$\alpha_0 = -0.1$', '$\alpha_0 = \frac{2}{3} \pi$'}, 'Location', 'best');
hold off

% Compute the analytical solution of beta
alpha1 = -acos((a(1)^2 + 2*a(1)*a(4)+a(2)^2-a(3)^2+a(4)^2)/(2*a(1)*a(2)+2*a(2)*a(4)));
alpha2 = acos((a(1)^2 + 2*a(1)*a(4)+a(2)^2-a(3)^2+a(4)^2)/(2*a(1)*a(2)+2*a(2)*a(4)));

% Plot the difference
ic1 = abs(alpha(1,1)-alpha1); 
ic2 = abs(alpha(1,2)-alpha2);
ic3 = abs(alphaFD(1,1)-alpha1);
ic4 = abs(alphaFD(1,2)-alpha2);

data1 = [ic1, ic3];
data2 = [ic2, ic4];
figure;
b = bar([1,3],data1, 'FaceColor', 'flat'); 
b.CData(1,:) = [0 0.4470 0.7410];  
b.CData(2,:) = [0.8500 0.3250 0.0980];   
xticks([1 3]); 
xticklabels({'Analytical', 'Finite Differences'}); 
title('Accuracy Comparison for $\alpha_0 = -0.1$');
xlabel('Methods');
ylabel('Mean Accuracy Difference');
grid on;

figure;
b = bar([1,3],data2, 'FaceColor', 'flat'); 
b.CData(1,:) = [0 0.4470 0.7410];  
b.CData(2,:) = [0.8500 0.3250 0.0980];   
xticks([1 3]);
xticklabels({'Analytical', 'Finite Differences'}); 
title('Accuracy Comparison for $\alpha_0 = \frac{2}{3} \pi$');
xlabel('Methods');
ylabel('Mean Accuracy Difference');
grid on;

% Repeat the procees for beta = [0 pi]

beta = linspace(0,pi,500);          % Definition of beta

n = max(size(beta));
m = max(size(alpha0));
alphaPi = zeros(n,m);
itPi = zeros(n,m);
realPi = zeros(n,m);

% Compute and the solution 
for j = 1:m
    for i = 1:n
        fun = @(alpha) f(alpha,beta(i));
        dfun = @(alpha) df(alpha,beta(i));
        realPi(i,j) = fzero(fun,alpha0(j));
        [alphaPi(i,j), itPi(i,j)] = NS(alpha0(j),nmax,toll,fun,dfun);
    end
end

% Plot the solutions
figure
plot(beta,alphaPi(:,1))
hold on
plot(beta,alphaPi(:,2))
xlim([0 3.5])
xlabel('$\beta$ [rad]');
ylabel('$\alpha$ [rad]');
title('NS solution of the analytical derivative')
legend('$\alpha_0 = -0.1$', '$\alpha_0 = \frac{2}{3} \pi$')
legend('Location','northwest')
hold off

figure
plot(beta,itPi(:,1))
hold on
plot(beta,itPi(:,2))
xlim([0 pi])
xlabel('$\beta$ [rad]');
ylabel('Iterations');
title('Iterations')
legend('$\alpha_0 = -0.1$', '$\alpha_0 = \frac{2}{3} \pi$')
legend('Location','northwest')
hold off

%% EXERCISE 2
clc
clear
close all


% Define initial data
v0 = 0;         %   Initial velocity        [m/s]
m0 = 20;        %   Initial mass            [kg]
cm = 0.1;       %   Decreasing coefficient  [kg/s]
f = 1;          %   External force          [N]
alpha = 0.01;   %   Velocity coefficient    [Ns/m]
Cd = 2.05;      %   Drag coefficient        [-]
Am = 1;         %   Area                    [m^2]
rho = 0;        %   Density                 [kg/m^3]
rho2 = 900;     %   Density                 [kg/m^3]

% Define the timespan
t0 = 0;
tmax = 160;
tex = t0:1:tmax;

% Define the step sizes
h = [50,20,10,1];

% Define the functions
vex = @(t) f/alpha - (f/alpha - v0).*(1-cm*t/m0).^(alpha/cm);

mfun = @(t) m0 - cm*t;
F = 0;
dvdt = @(t,v) (F + f -alpha*v)/mfun(t);

% Solve the functions and compare with exact solution
n = length(h);

[v1, th1] = Heun(dvdt,t0,tmax,v0,h(1));

[v2, th2] = Heun(dvdt,t0,tmax,v0,h(2));

[v3, th3] = Heun(dvdt,t0,tmax,v0,h(3));

[v4, th4] = Heun(dvdt,t0,tmax,v0,h(4));

% Create a figure
set(groot, 'defaultLegendFontSize', 10,'DefaultLegendFontSizeMode','manual');
figure;
subplot(2, 2, 1);
plot(th1, v1);
hold on
plot(th1,vex(th1),"LineStyle","--",LineWidth=1.5)
hold off
title('h = 50');
xlim([0 160])
xlabel('time [s]');
ylabel('velocity [m/s]');
legend('h = 50','Exact','Location','northwest')
grid on;

subplot(2, 2, 2);
plot(th2, v2);
hold on
plot(th2,vex(th2),"LineStyle","--",LineWidth=1.5)
hold off
title('h = 20');
xlabel('time [s]');
xlim([0 160])
ylabel('velocity [m/s]');
legend('h = 20','Exact','Location','northwest')
grid on;

subplot(2, 2, 3);
plot(th3, v3);
hold on
plot(th3,vex(th3),"LineStyle","--",LineWidth=1.5)
hold off
title('h = 10');
xlim([0 160])
xlabel('time [s]');
ylabel('velocity [m/s]');
legend('h = 10','Exact','Location','northwest')
grid on;

subplot(2, 2, 4);
plot(th4, v4);
hold on
plot(th4,vex(th4),"LineStyle","--",LineWidth=1.5)
hold off
title('h = 1');
xlim([0 160])
xlabel('time [s]');
ylabel('velocity [m/s]');
legend('h = 1','Exact','Location','northwest')
grid on;

sgtitle('Heun Method comparison for different step sizes'); % Title for the entire figure

% Compute the error 
figure;
subplot(2, 2, 1);
plot(th1, abs(v1-vex(th1)));
title('h = 50');
xlim([0 160])
xlabel('time [s]');
ylabel('error [m/s]');
grid on;

subplot(2, 2, 2);
plot(th2, abs(v2-vex(th2)));
title('h = 20');
xlim([0 160])
xlabel('time [s]');
ylabel('error [m/s]');
grid on;

subplot(2, 2, 3);
plot(th3, abs(v3-vex(th3)));
title('h = 10');
xlim([0 160])
xlabel('time [s]');
ylabel('error [m/s]');
grid on;

subplot(2, 2, 4);
plot(th4, abs(v4-vex(th4)));
title('h = 1');
xlim([0 160])
xlabel('time [s]');
ylabel('error [m/s]');
grid on;

sgtitle('Error comparison for different step sizes'); % Title for the entire figure

set(groot, 'defaultLegendFontSize', 12,'DefaultLegendFontSizeMode','manual');

figure
plot(th1, abs(v1-vex(th1)));
hold on
plot(th2, abs(v2-vex(th2)));
plot(th3, abs(v3-vex(th3)));
plot(th4, abs(v4-vex(th4)));
hold off
title('Error comparison for different step sizes');
xlim([0 160])
xlabel('time [s]');
ylabel('error [m/s]');
legend('h = 50','h = 20', 'h = 10', 'h = 1', 'Location','northwest')

% Repeat the process with Runge Kutta 4

[vRK1, thRK1] = RK4(dvdt,t0,tmax,v0,h(1));

[vRK2, thRK2] = RK4(dvdt,t0,tmax,v0,h(2));

[vRK3, thRK3] = RK4(dvdt,t0,tmax,v0,h(3));

[vRK4, thRK4] = RK4(dvdt,t0,tmax,v0,h(4));


% Create a figure
set(groot, 'defaultLegendFontSize', 10,'DefaultLegendFontSizeMode','manual');
figure;
subplot(2, 2, 1);
plot(thRK1, vRK1);
hold on
plot(thRK1,vex(thRK1),"LineStyle","--",LineWidth=1.5)
hold off
title('h = 50');
xlim([0 160])
xlabel('time [s]');
ylabel('velocity [m/s]');
legend('h = 50','Exact','Location','northwest')
grid on;

subplot(2, 2, 2);
plot(thRK2, vRK2);
hold on
plot(thRK2,vex(thRK2),"LineStyle","--",LineWidth=1.5)
hold off
title('h = 20');
xlabel('time [s]');
xlim([0 160])
ylabel('velocity [m/s]');
legend('h = 20','Exact','Location','northwest')
grid on;

subplot(2, 2, 3);
plot(thRK3, vRK3);
hold on
plot(thRK3,vex(thRK3),"LineStyle","--",LineWidth=1.5)
hold off
title('h = 10');
xlim([0 160])
xlabel('time [s]');
ylabel('velocity [m/s]');
legend('h = 10','Exact','Location','northwest')
grid on;

subplot(2, 2, 4);
plot(thRK4, vRK4);
hold on
plot(thRK4,vex(thRK4),"LineStyle","--",LineWidth=1.5)
hold off
title('h = 1');
xlim([0 160])
xlabel('time [s]');
ylabel('velocity [m/s]');
legend('h = 1','Exact','Location','northwest')
grid on;

sgtitle('Runge-Kutta 4 Method comparison for different step sizes'); % Title for the entire figure

% Compute the error 
figure;
subplot(2, 2, 1);
plot(thRK1, abs(vRK1-vex(thRK1)));
title('h = 50');
xlim([0 160])
xlabel('time [s]');
ylabel('error [m/s]');
grid on;

subplot(2, 2, 2);
plot(thRK2, abs(vRK2-vex(thRK2)));
title('h = 20');
xlim([0 160])
xlabel('time [s]');
ylabel('error [m/s]');
grid on;

subplot(2, 2, 3);
plot(thRK3, abs(vRK3-vex(thRK3)));
title('h = 10');
xlim([0 160])
xlabel('time [s]');
ylabel('error [m/s]');
grid on;

subplot(2, 2, 4);
plot(thRK4, abs(vRK4-vex(thRK4)));
title('h = 1');
xlim([0 160])
xlabel('time [s]');
ylabel('error [m/s]');
grid on;

sgtitle('Error comparison for different step sizes'); % Title for the entire figure

set(groot, 'defaultLegendFontSize', 12,'DefaultLegendFontSizeMode','manual');

figure
plot(thRK1, abs(vRK1-vex(thRK1)));
hold on
plot(thRK2, abs(vRK2-vex(thRK2)));
plot(thRK3, abs(vRK3-vex(thRK3)));
plot(thRK4, abs(vRK4-vex(thRK4)));
hold off
title('Error comparison for different step sizes');
xlim([0 160])
xlabel('time [s]');
ylabel('error [m/s]');
legend('h = 50','h = 20', 'h = 10', 'h = 1', 'Location','northwest')

% Trade-off cpu time and integration error

iter = 1000;

tcpuH = zeros(n,iter);
errH = zeros(n,iter);
tcpuRK = zeros(n,iter);
errRK = zeros(n,iter);

avgErrH = zeros(1,n);
varErrH = zeros(1,n);
avgErrRK = zeros(1,n);
varErrRK = zeros(1,n);

avgTH = zeros(1,n);
varTH = zeros(1,n);
avgTRK = zeros(1,n);
varTRK = zeros(1,n);

for i = 1:n
    for j = 1:iter
        tstart = tic;
        [v, th] = Heun(dvdt,t0,tmax,v0,h(i));
        tcpuH(i,j) = toc(tstart);
        errH(i,j) = max(abs(v-vex(th)));
    end
    avgTH(i) = mean(tcpuH(i,:));
    varTH(i) = var(tcpuH(i,:));
    avgErrH(i) = mean(errH(i,:));
    varErrH(i) = var(errH(i,:));
end

for i = 1:n
    for j = 1:iter
        tstart = tic;
        [v, th] = RK4(dvdt,t0,tmax,v0,h(i));
        tcpuRK(i,j) = toc(tstart);
        errRK(i,j) = max(abs(v-vex(th)));
    end
    avgTRK(i) = mean(tcpuRK(i,:));
    varTRK(i) = var(tcpuRK(i,:));
    avgErrRK(i) = mean(errRK(i,:));
    varErrRK(i) = var(errRK(i,:));
end

figure;
bar(avgTH, 'FaceColor', [0.2 0.6 0.8],'BarWidth', 0.4); 
hold on;
errorbar(1:4, avgTH, sqrt(varTH), 'Color',[0 0 0.5], 'LineStyle', 'none', 'LineWidth', 1.5);
xlabel('Stepsize');
ylabel('CPU Time Average');
title('Comparison of CPU time for Heun Method');
xticks(1:4); 
xticklabels({'h = 50', 'h = 20', 'h = 10', 'h = 1'});

figure;
bar(avgTRK, 'FaceColor', [0.2 0.6 0.8],'BarWidth', 0.4); 
hold on;
errorbar(1:4, avgTRK, sqrt(varTRK), 'Color',[0 0 0.5], 'LineStyle', 'none', 'LineWidth', 1.5);
xlabel('Stepsize');
ylabel('CPU Time Average');
title('Comparison of CPU time for RK4 Method');
xticks(1:4); 
xticklabels({'h = 50', 'h = 20', 'h = 10', 'h = 1'});

figure;
bar(avgErrH, 'FaceColor', [0.2 0.6 0.8],'BarWidth', 0.4); 
hold on;
xlabel('Stepsize');
ylabel('Maximum Error Mean');
title('Maximum error for Heun Method');
xticks(1:4); 
xticklabels({'h = 50', 'h = 20', 'h = 10', 'h = 1'});

figure;
bar(avgErrRK, 'FaceColor', [0.2 0.6 0.8],'BarWidth', 0.4); 
hold on;
xlabel('Stepsize');
ylabel('Maximum Error Mean');
title('Maximum error for RK4 Method');
xticks(1:4); 
xticklabels({'h = 50', 'h = 20', 'h = 10', 'h = 1'});

% Repeat the process with a non-zero drag

F1 = @(v) -0.5*rho2*Cd*Am*v^2;
dvdt1 = @(t,v) (F1(v) + f -alpha*v)/mfun(t);

t0 = 0;
tmax = 160;
tspan = t0:h(4):tmax;

[v1, th1] = Heun(dvdt1,t0,tmax,v0,h(4));

figure
plot(th1,v1)
xlabel('time [s]');
ylabel('velocity [m/s]');
title('Heun Solutions with non-zero drag')

% Solve with proper matlab ode
[~, vode] = ode15s(dvdt1,tspan,v0);

figure
plot(tspan,vode)
xlabel('time [s]');
ylabel('velocity [m/s]');
title('ODE15s Solutions with non-zero drag')

% Plot stability region
tol = 1e-12;
lambda(1) = -0.0005;    %   rho = 0 kg/m^3
lambda(2) = -3.0373;    %   rho = 900 kg/m^3


stabilityPlot('RK2',tol)
hold on
p2 = plot(real(lambda(2)), imag(lambda(2)), 'x', 'MarkerSize', 12, 'LineWidth', 1.5);
p1 = plot(real(lambda(1)), imag(lambda(1)), 'x', 'MarkerSize', 12, 'LineWidth', 1.5);
legend([p1,p2],{'$\rho = 0 \ kg/m^3$', '$\rho = 900 \ kg/m^3$'}, 'Location','northwest')
hold off
                         
%% EXERCISE 3
clc
clear
close all

% Define the operators
A = @(a) [0, 1; -1, 2*cos(a)];
I = eye(2);

% Define initial parameters
ah = pi;
FRK2 = @(h,a) I + A(a)*h + (A(a)*h)^2/2;

% Solve the problem a = pi
Prob = @(h) max(abs(eig(FRK2(h,ah)))) - 1;
h0 = ah;
hRK2 = fzero(Prob,h0);

% Repeat for RK2 for alpha 0:pi
tol = 1e-12;
stabilityPlot('RK2',tol)

% Repeat for RK4
ah = pi;
FRK4 = @(h,a) I + A(a)*h + (A(a)*h)^2/2 + (A(a)*h)^3/6 + (A(a)*h)^4/24;

% Solve the problem a = pi
Prob = @(h) max(abs(eig(FRK4(h,ah)))) - 1;
h0 = ah;
hRK4 = fzero(Prob,h0);

% Repeat for RK4 for alpha 0:pi
tol = 1e-12;
figure
stabilityPlot('RK4',tol)

% Repeat with backinterpolation methods

% Theta = 0.3
theta = 0.3;
tol = 1e-6;
stabilityPlot('BI2',tol,theta)

% Repeat with variable thetas
theta = [0.2, 0.4, 0.6, 0.8];
j = max(size(theta));

for i = 1:j
    stabilityPlot('BI2',tol,theta(i))
end

%% EXERCISE 4
clc
clear
close all

% Define initial variables
Kc = 0.0042;                % Convective heat loss coefficient    [J/(s K)]
Kr = 6.15e-11;              % Radiative heat loss coefficient     [J/(s K^4)]
C = 45;                     % Mass thermal capacity               [J/K]
Ta = 277;                   % Air temperature                     [K]
T0 = 555;                   % Initial mass temperature            [K]

t0 = 0;
tmax = 72000;


% Define the function
dTdt = @(t, T) -Kc*(T-Ta)/C - Kr*(T^4 - Ta^4)/C;

% Solve with RK2
hRK2 = 720;          % RK2 time step
[TH,tH] = Heun(dTdt,t0,tmax,T0,hRK2);

% Solve with RK4
hRK4 = 1440;         % RK4 time step
[TRK,tRK] = RK4(dTdt,t0,tmax,T0,hRK4);

% Solve with teh proper Matlab ODE
[tode, Tode] = ode78(dTdt,tH,T0);
[todeRK, TodeRK] = ode78(dTdt,tRK,T0);

% Compute equilibrium timestamps
tol = 1e-3;

iH = find(TH-Ta<tol,1);
eqH = tH(iH);

iRK = find(TRK-Ta<tol,1);
eqRK = tRK(iRK);

iOde = find(Tode-Ta<tol,1);
eqOde = tode(iOde);

% Plot the results

figure;
plot(tH/3600,TH)
hold on
yline(Ta,'LineStyle','--','LineWidth',1.5,'Color',[0.3 0.3 0.3])
xlabel('Time [h]');
ylabel('Temperature [K]');
legend('RK2','$T_{air}$', 'Location','northeast')

figure;
plot(tRK/3600,TRK,Color=[0.8500, 0.3250, 0.0980])
hold on
yline(Ta,'LineStyle','--','LineWidth',1.5,'Color',[0.3 0.3 0.3])
xlabel('Time [h]');
ylabel('Temperature [K]');
legend('RK4','$T_{air}$', 'Location','northeast')

figure
subplot(2,1,1)
plot(tH/3600,TH)
hold on
plot(tode/3600, Tode,LineStyle="--",Color=[0.8500, 0.3250, 0.0980])
yline(Ta,'LineStyle','--','LineWidth',1.5,'Color',[0.3 0.3 0.3])
title('Comparison between RK2 and \texttt{ode78}')
ylim([250 600])
ylabel('Temperature [K]');
legend('RK2','ode78','$T_{air}$', 'Location','northeast')

subplot(2,1,2)
plot(tH/3600,abs(TH-Tode'))
title('Error between RK2 and \texttt{ode78}')
xlabel('Time [h]');
ylabel('Error [K]');

figure
subplot(2,1,1)
plot(tRK/3600,TRK)
hold on
plot(todeRK/3600, TodeRK,LineStyle="--",Color=[0.8500, 0.3250, 0.0980])
yline(Ta,'LineStyle','--','LineWidth',1.5,'Color',[0.3 0.3 0.3])
title('Comparison between RK4 and \texttt{ode78}')
ylim([250 600])
ylabel('Temperature [K]');
legend('RK4','ode78','$T_{air}$', 'Location','northeast')

subplot(2,1,2)
plot(tRK/3600,abs(TRK-TodeRK'))
title('Error between RK4 and \texttt{ode78}')
xlabel('Time [h]');
ylabel('Error [K]');

%% EXERCISE 5
clc
clear
close all

% Define initial parameters
R = 25;             % Resistance [Ohm]
L = 0.02;           % Inductance [H]
C = 0.2;            % Capacitance [F]
v0 = 12;            % Capacistor voltage [V]
tol = 1e-6;         % Solver tolerances

% Define the state space model
dxdt = @(t, x) [x(2); -1/(L*C)*x(1) - R/L*x(2)];

% Initial conditions
x0 = [v0 * C; 0];  

% Time span for simulation
t0 = 0;
tmax = 30;

% Define the A
Aprob = [0 1; -1/(L*C) -R/L];
lambdaA = eig(Aprob);
% First eig
Re1 = real(lambdaA(1));
Im1 = imag(lambdaA(1));
% Second eig
Re2 = real(lambdaA(2));
Im2 = imag(lambdaA(2));

set(groot, 'defaultLegendFontSize', 10,'DefaultLegendFontSizeMode','manual');

% Plot eig and RK2 stability
figure
subplot(1, 2, 1);           % Farthest eigenvalue  
hold on
plot(Re2, Im2, 'bx', 'MarkerSize', 10)
yline(0, 'Color', [0.3 0.3 0.3], 'LineWidth', 0.75); % x-axis
axis equal
xlabel('$Re \{ h \lambda \}$')
ylabel('$Im \{ h \lambda \}$')
title(' Eigenvalue $\lambda_2$')
xlim([-1251, -1249]);
ylim([-2 2]);

subplot(1, 2, 2);            % Closest eigenvalue                                           
plot(Re1, Im1, 'bx', 'MarkerSize', 10)
hold on
stabilityPlot('RK2', tol);
legend('Eigenvalue', 'RK2 Stability Region')
hold off

%  IEX4 stability region
xLimits = [-1255, -1243];  
yLimits = [-10 10]; 

figure
subplot(1, 2, 1);              % Farthest eigenvalue 
fill([xLimits(1), xLimits(1), xLimits(2), xLimits(2)],[yLimits(1), yLimits(2), yLimits(2), yLimits(1)],[0.7 0.9 1], 'FaceAlpha', 0.5, 'EdgeColor', 'none');    
hold on
plot(Re2, Im2, 'rx', 'MarkerSize', 10)
yline(0, 'Color', [0.3 0.3 0.3], 'LineWidth', 0.75); % x-axis
axis equal
xlabel('$Re \{ h \lambda \}$')
ylabel('$Im \{ h \lambda \}$')
title(' Eigenvalue $\lambda_2$')
xlim([-1255, -1243]);
ylim([-10 10]);

subplot(1, 2, 2);               % Closest eigenvalue                                              
plot(Re1, Im1, 'rx', 'MarkerSize', 10)
hold on
stabilityPlot('IEX4');
axis equal
legend('Eigenvalue', 'IEX4 Stable Region')
xlim([-3, 15]);
ylim([-15 15]);

set(groot, 'defaultLegendFontSize', 12,'DefaultLegendFontSizeMode','manual');


% Solve the system using IEX4
h = 0.1;
[xIEX4,tIEX4] = IEX4(Aprob,t0,tmax,x0,h);

figure
plot(tIEX4, xIEX4(1,:), 'color', [0.2, 0.8, 0.2]);
xlabel('Time (s)');
ylabel('Charge (q)');
title('Charge on the Capacitor using IEX4');

figure()
plot(tIEX4, xIEX4(2,:), 'color', [0.9, 0.6, 0.1]);
xlabel('Time (s)');
ylabel('Current (i)');
title('Current in the Circuit using IEX4');

% Find the maximum h for RK2
hMax = (-2/lambdaA(2)) - 0.0001;                  

fun = @(t,x) Aprob*x;

[xRK2, tRK2] = Heun(fun,t0,tmax,x0,hMax);

figure
plot(tRK2, xRK2(1,:), 'color', [0.2, 0.8, 0.2]);
xlabel('Time (s)');
ylabel('Charge (q)');
title('Charge on the Capacitor using RK2');

figure
plot(tRK2, xRK2(2,:), 'color', [0.9, 0.6, 0.1]);
xlabel('Time (s)');
ylabel('Current (i)');
title('Current in the Circuit using RK2');

% Runtime comparison
Time = zeros(2, 100);

for i = 1 : max(size(Time))
    % Heun function runtime
    tic
    [~, ~] = IEX4(Aprob,t0,tmax,x0,h);
    Time(1, i) = toc;
    % IEX4 function runtime
    tic
    [~, ~] = Heun(fun,t0,tmax,x0,hMax);
    Time(2, i) = toc;
end

MeanTime = mean(Time, 2);

%% EXERCISE 6
clc
clear
close all

% Define initial variables
rho = 1.225;        % Air density [kg/m^3]
mb = 1;             % Ball mass [kg]
Ab = 0.07;          % Ball area [m^2]
Vb = 0.014;         % Ball volume [m^3]
Cd = 1.17;          % Drag coefficient [-]
g = 9.81;           % Gravity accelertion [m/s^2]
k = 0.9;            % Attenuation factor [-]

% Define the time span
t0 = 0;
tmax = 10;

% Initial conditions
v0 = 0;                   
x0 = 10;                   
y0 = [v0; x0];  

% Forces
FD = @(v) -0.5 * rho * Cd * Ab * v * abs(v); 
FA = rho * Vb * g;                      

% Differential equations
dydt = @(t,y) [-g + (FD(y(1)) + FA)/mb; y(1)];

% Solve ODE with event function for ground impact
tsol = [];
ysol = [];

tstart = t0;
ystart = y0;

while tstart < tmax
    options = odeset('Events', @groundEvent, 'RelTol', 1e-9, 'AbsTol', 1e-9);
    [ts, ys, te, ye, ie] = ode45(dydt, [tstart tmax], ystart, options);

    tsol = [tsol; ts(2:end)];  
    ysol = [ysol; ys(2:end,:)]; 

    if isempty(te)
        break;  
    end

    tstart = te;                 
    ystart = [ -k * ye(1); 0 ];  
end

% Plot the results
figure;
plot(tsol, ysol(:,2),'Color', [0.9, 0.6, 0.1]); 
xlabel('Time [s]');
ylabel('Position [m]');
title('Position of the Bouncing Ball');
ylim([0 10])

figure
plot(tsol, ysol(:,1)); 
hold on
yline(0, '--k');
xlabel('Time [s]');
ylabel('Velocity [m/s]');
title('Velocity of the Bouncing Ball');
ylim([-12 11])

% Repeat the problem for rho = 15
rho2 = 15;

FD2 = @(v) -0.5 .* rho2 .* Cd .* Ab .* v .* abs(v); 
FA2 = rho2 * Vb * g;  

dydt2 = @(t,y) [-g + (FD2(y(1)) + FA2)/mb; y(1)];

tsol2 = [];
ysol2 = [];

tstart = t0;
ystart = y0;

while tstart < tmax
    options = odeset('Events', @groundEvent, 'RelTol', 1e-9, 'AbsTol', 1e-9);
    [ts, ys, te, ye, ie] = ode45(dydt2, [tstart tmax], ystart, options);
    
    tsol2 = [tsol2; ts(2:end)];  
    ysol2 = [ysol2; ys(2:end,:)]; 

    if abs(ys(end,1)) < 1e-6
        break;  
    end

    if isempty(te)
        break;  
    end

    tstart = te;                 
    ystart = [ -k * ye(1); 0 ];  
end

figure;
plot(tsol2, ysol2(:,2), 'Color', [0.9, 0.6, 0.1]); 
xlabel('Time [s]');
ylabel('Position [m]');
title('Position of the Bouncing Ball');
ylim([0 10])

figure,
plot(tsol2, ysol2(:,1)); 
hold on
yline(0, '--k');
xlabel('Time [s]');
ylabel('Velocity [m/s]');
title('Velocity of the Bouncing Ball');

% Repeat the problem for rho = 60
rho3 = 60;

FD3 = @(v) -0.5 * rho3 * Cd * Ab * v * abs(v); 
FA3 = rho3 * Vb * g;  

dydt3 = @(t,y) [-g + (FD3(y(1)) + FA3)/mb; y(1)];

tsol3 = [];
ysol3 = [];

tstart = t0;
ystart = y0;

while tstart < tmax
    options = odeset('Events', @groundEvent, 'RelTol', 1e-10, 'AbsTol', 1e-10);
    [ts, ys, te, ye, ie] = ode45(dydt3, [tstart tmax], ystart, options);
    
    tsol3 = [tsol3; ts(2:end)];  
    ysol3 = [ysol3; ys(2:end,:)]; 

    if abs(ys(end,1)) < 1e-6
        break;  
    end

    if isempty(te)
        break;  
    end

    tstart = te;                 
    ystart = [ -k * ye(1); 0 ];  
end

figure;
plot(tsol3, ysol3(:,2), 'Color', [0.9, 0.6, 0.1]); 
xlabel('Time [s]');
ylabel('Position [m]');
title('Position of the Bouncing Ball');
ylim([0 10])

figure;
plot(tsol3, ysol3(:,1)); 
hold on
yline(0, '--k');
xlabel('Time [s]');
ylabel('Velocity [m/s]');
title('Velocity of the Bouncing Ball');


% Solve the problem using RK4 with h = 3
h = 3;
tspan = t0:h:tmax;

[yRK4, tRK4] = RK4(dydt3, t0, tmax, y0, h);

%  Plot 
%  Stability Region
figure;
plot(0, 0, 'bx', 'MarkerSize', 10); 
hold on
stabilityPlot('RK4');
legend('$\lambda_1 = \lambda_2$','RK4 stability region', 'location', 'northwest')
title('RK4 Method')

figure;
subplot(2, 1, 1);
plot(tRK4, yRK4(2,:) ,'Color', [0.9, 0.6, 0.1]); 
xlabel('Time [s]');
ylabel('Position [m]');
title('Position of the Bouncing Ball');
grid on;
ylim([0 10])

% Velocity plot
subplot(2, 1, 2);
plot(tRK4, yRK4(1,:), '-'); 
hold on
yline(0, '--k');
xlabel('Time [s]');
ylabel('Velocity [m/s]');
title('Velocity of the Bouncing Ball');
grid on;

% repeat for h = 0.5
h2 = 0.5;

[yRK4, tRK4] = RK4(dydt3, t0, tmax, y0, h2);

%  Plot 
figure;
subplot(2, 1, 1);
plot(tRK4, yRK4(2,:) ,'Color', [0.9, 0.6, 0.1]); 
xlabel('Time [s]');
ylabel('Position [m]');
title('Position of the Bouncing Ball');
grid on;
ylim([0 10])

% Velocity plot
subplot(2, 1, 2);
plot(tRK4, yRK4(1,:), '-'); % Velocity over time
hold on
yline(0, '--k');
xlabel('Time [s]');
ylabel('Velocity [m/s]');
title('Velocity of the Bouncing Ball');
grid on;

%% FUNCTIONS


function [xs,it]=NS(x0,nmax,toll,fun,dfun,mol)

% [xs,it]=NS(x0,nmax,toll,fun,dfun) 
%
% Newton Solver 
%
% INPUT:
% x0            Initial guess 
% nmax          Maximum iterations number
% toll          Tollerance 
% fun           Function to solve
% dfun          First derivative of the function 
% mol           Modified Newton method
%
% OUTPUT:
% xs            Solution of the problem   
% it            Number of iterations used

if (nargin == 5)
    mol = 1;
end

err = toll + 1;
it = 0;
xv = x0;
xit = x0;

while (it< nmax && err> toll)
   dfx = dfun(xit);
   if abs(dfx)< toll
      %error('First derivative equal to zero');
      xs = NaN;
      return
   else
      xn = xit - mol*fun(xit)/dfx;
      err = abs(xn-xit);
      xv = [xv; xn];
      it = it+1;
      xit = xn;
   end
end

if(it == nmax)
    xs = NaN;
    return
end

xs = xv(end);
end

%--------------------------------------------------------------------------------

function [xs,it]=NS_FD(x0,nmax,toll,fun,h)

% [xs,it]=NS_FD(x0,nmax,toll,fun,h) 
%
% Newton Solver which uses Finite Difference method to compute the
% derivative of the function.
%
% INPUT:
% x0            Initial guess 
% nmax          Maximum iterations number
% toll          Tollerance 
% fun           Function to solve
% h             Step size
%
% OUTPUT:
% xs            Solution of the problem   
% it            Number of iterations used


err = toll + 1;
it = 0;
xv = x0;
xit = x0;

while (it< nmax && err> toll)
   dfx = (fun(xit+h)-fun(xit-h))/(2*h);
   if abs(dfx)< toll
      error('First derivative equal to zero');
   else
      xn = xit - fun(xit)/dfx;
      err = abs(xn-xit);
      xv = [xv; xn];
      it = it+1;
      xit = xn;
   end
end

xs = xv(end);
end

%-----------------------------------------------------------------------------

function  [xv,th]=Heun(fun,t0,tmax,y0,h)

% [xv,th]=Heun(fun,t0,tmax,y0,h)
%
% Risolve il problema utilizzando il metodo di Heun: 
% u_{n+1}=u_n+1/2*h*( f(t_n,u_n) +f(t_{n+1},u_n+h f(t_n,u_n))). 
%
% INPUT:
% fun           Function to solve 
% t0            Initial Timestep
% tmax          Final istant of the temporal interval
% y0            Initial guess 
% h             Step size
%
% OUTPUT:
% xv            Vector containing the solution of the problem   
% th            Time vector

th = t0:h:tmax;

StepN = length(th);
varN = length(y0);

xv = zeros(varN,StepN);

xv(:,1) = y0;

for it = 2:StepN
    xv(:,it) = xv(:,it-1) + h/2 * ( fun(th(it-1),xv(:,it-1)) + fun(th(it),xv(:,it-1) + h*fun(th(it-1),xv(:,it-1))) ); 
end

end

%--------------------------------------------------------------------------
function [xv, th] = RK4(fun, t0, tmax, y0, h)

% [xv, th] = RK4(fun, t0, tmax, y0, h)
%
% Solves a system of ODEs using the 4th-order Runge-Kutta method.
%
% INPUT:
% fun     - Function handle representing the system (dy/dt = fun(t, y))
% t0      - Initial time
% tmax    - Final time
% y0      - Initial state vector
% h       - Step size
%
% OUTPUT:
% xv      - Solution matrix where each column is the state at a specific time step
% th      - Vector of time steps

th = t0:h:tmax;
StepN = length(th);

numEq = length(y0);    
xv = zeros(numEq, StepN); 
xv(:, 1) = y0;             

for i = 2:StepN
    ti = th(i - 1);
    yi = xv(:, i - 1);

    k1 = fun(ti, yi);
    k2 = fun(ti + 0.5 * h, yi + 0.5 * h * k1);
    k3 = fun(ti + 0.5 * h, yi + 0.5 * h * k2);
    k4 = fun(ti + h, yi + h * k3);

    xv(:, i) = yi + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
end

end

%-----------------------------------------------------------------------------

function  [xv,th]=IEX4(A,t0,tmax,y0,h)

% [xv,th]=IEX4(A,t0,tmax,y0,h)
%
% Solves the problem using implicit extrapolation techniques
%
% INPUT:
% A             A matrix for a linear system f = A*x 
% t0            Initial Timestep
% tmax          Final istant of the temporal interval
% y0            Initial guess 
% h             Step size
%
% OUTPUT:
% xv            Vector containing the solution of the problem   
% th            Time vector

th=t0:h:tmax;

StepN = length(th);
xv=zeros(2,StepN);
I = eye(2);

xv(:,1)=y0;

for i=2:StepN
    k1 = (I-h.*A)^(-1)*xv(:,i-1);
    k2 = (I-h/2.*A)^(-2)*xv(:,i-1);
    k3 = (I-h/3.*A)^(-3)*xv(:,i-1);
    k4 = (I-h/4.*A)^(-4)*xv(:,i-1);

    xv(:,i) = -1/6*k1 + 4*k2 - 27/2*k3 + 32/3*k4;
end

end 

%-----------------------------------------------------------------------------

function [value, isterminal, direction] = groundEvent(~, y)

% [value, isterminal, direction] = groundEvent(~, y)
%
% Event function which changes the velocity of a ball when hitting the
% gound bouncing.
%
% INPUT:
% y             State space vector where y(1) = velocity and y(2) = position
%
% OUTPUT:
% value         Detects when the ball hits the ground    
% isterminal    Stops the integration at the event detection
% direction     Detect only downward events (ball hitting the ground)


    value = y(2);         
    isterminal = 1;      
    direction = -1;       
    
end

%-----------------------------------------------------------------------------
function stabilityPlot(method,tol,theta)

% stabilityPlot(method,F,tol,theta)
%
% Plots the stability region of the problem given an itial alpha value 
%
% INPUT:
% method        Char containing the name of the method
% tol           Tollerance to consider h as         (optional) 
% theta         parameter for BI2 method            (optional)
%
% OUTPUT:
%               It plots the stability region of the method

if nargin < 3, theta = NaN; end
if nargin < 2, tol = 1e-6; end

%   System matrix with a pair of complex conjugate eigenvalues 
%   on the unit circle; alpha defines the angle from the real axis

A = @(alpha) [0 1; -1 2*cos(alpha)];
I = eye(2);

switch method
    case 'RK2'
        
        n = 100;
        vecAlpha = linspace(0,pi,n);
        F = @(h,a) I + A(a)*h + (A(a)*h)^2/2;

        h0 = vecAlpha;
        [~,Re1,Im1,Re2,Im2] =  Solver(A,F,vecAlpha,h0,tol);

        
        %   Plot
        fill(Re1(~isnan(Re1)), Im1(~isnan(Im1)), 'r', 'FaceAlpha', 0.25, 'EdgeColor','none') 
        hold on
        fill(Re2(~isnan(Re2)), Im2(~isnan(Im2)), 'r', 'FaceAlpha', 0.25, 'EdgeColor','none')
        plot(Re1,Im1,'r')
        plot(Re2,Im2,'r')
        xline(0, 'Color', [0.3 0.3 0.3], 'LineWidth', 0.75); % x-axis
        yline(0, 'Color', [0.3 0.3 0.3], 'LineWidth', 0.75); % y-axis
        axis equal
        title('RK2 stability region')
        xlabel('$Re \{ h \lambda \}$')
        ylabel('$Im \{ h \lambda \}$')
        legend('Stable')
        hold off

    case 'RK4'

        % To fully simulate the stability region of the RK4 method multiple
        % alpha vectors are created with different initial conditions

        F = @(h,a) I + A(a)*h + (A(a)*h)^2/2 + (A(a)*h)^3/6 + (A(a)*h)^4/24;

        %   Alpha vectors
        nAlphaA = 1e3;
        nAlphaB = 1e2;
        vecAlphaA = linspace(0,pi/2, nAlphaA);
        vecAlphaB = linspace(pi/2,pi, nAlphaB);
        
        %   Initual guesses vectors
        guessA = linspace(2,4, nAlphaA);
        guessA2 = linspace(1.4,1.6, nAlphaA);
        guessB = linspace(2, 5, nAlphaB);
        
        [~,Re1A,Im1A,Re2A,Im2A] = Solver(A,F,vecAlphaA,guessA,tol);
        [~,Re1A2,Im1A2,Re2A2,Im2A2] = Solver(A,F,vecAlphaA,guessA2,tol);
        [~,Re1B,Im1B,Re2B,Im2B] = Solver(A,F,vecAlphaB,guessB,tol);
      
        
        %   Remove spourius points
        Re1A(abs(Im1A) < tol) = NaN;
        Im1A(abs(Im1A) < tol) = NaN;
        Re2A(abs(Im2A) < tol) = NaN;
        Im2A(abs(Im2A) < tol) = NaN;

        Re1A2(abs(Im1A2) < tol) = NaN;
        Im1A2(abs(Im1A2) < tol) = NaN;
        Re2A2(abs(Im2A2) < tol) = NaN;
        Im2A2(abs(Im2A2) < tol) = NaN;
        
        [~, maxID] = max(Im1A2);
        Im1A2(maxID) = NaN;
        Re2A2(maxID) = NaN;
        
        
        [~, maxID] = max(abs(Im2A2));
        Im2A2(maxID) = NaN;
        Re2A2(maxID) = NaN;
        
        %   Rewrite vectors for plot
        hLRe1 = [0,flip(Re1A2(~isnan(Re1A2))), Re1A(~isnan(Re1A)), Re1B(~isnan(Re1B))];
        hLIm1 = [0,flip(Im1A2(~isnan(Im1A2))), Im1A(~isnan(Im1A)), Im1B(~isnan(Im1B))];
        hLRe2 = [0,flip(Re2A2(~isnan(Re2A2))), Re2A(~isnan(Re2A)), Re2B(~isnan(Re2B))];
        hLIm2 = [0,flip(Im2A2(~isnan(Im2A2))), Im2A(~isnan(Im2A)), Im2B(~isnan(Im2B))];

        hLRe1 = hLRe1(2:end);
        hLRe1(1) = 0;
        
        %   Plot
        fill(hLRe1, hLIm1, 'r', 'FaceAlpha', 0.25, 'EdgeColor','none') 
        hold on
        fill(hLRe2, hLIm2, 'r', 'FaceAlpha', 0.25, 'EdgeColor','none')
        plot(hLRe1, hLIm1, 'r')
        plot(hLRe2, hLIm2, 'r')
        axis equal
        xline(0, 'Color', [0.3 0.3 0.3], 'LineWidth', 0.75); % x-axis
        yline(0, 'Color', [0.3 0.3 0.3], 'LineWidth', 0.75); % y-axis
        title('RK4 stability regions')
        xlabel('$Re \{ h \lambda \}$')
        ylabel('$Im \{ h \lambda \}$')
        legend('Stable')  
        hold off

    case 'BI2'
        %   BI2 operator
        BBI2 = @(h,a,th) (I - A(a)*(1-th)*h + (A(a)*(1-th)*h)^2/2)^(-1) * (I + A(a)*th*h + (A(a)*th*h)^2/2);
        F = @(h,a) BBI2(h,a,theta);

        switch theta
            case 0.2
                % Alpha vector
                nAlpha = 100;
                vecAlpha = linspace(0, pi, nAlpha);

                % Guess vector
                guess = -1.*linspace(0, pi, nAlpha);

                % Axis limit for visualization
                AxLimX = [-4 4];
                AxLimY = [-4 4];
                
                % Solution
                [~, Re1, Im1, Re2, Im2] = Solver(A,F,vecAlpha,guess,tol);

            case 0.3
                % Alpha vector
                nAlpha = 100;
                vecAlpha = linspace(0, pi, nAlpha);

                % Guess vector
                guess = -1.*linspace(0, pi, nAlpha);

                % Axis limit for visualization
                AxLimX = [-6 6];
                AxLimY = [-6 6];

                % Solution
                [~, Re1, Im1, Re2, Im2] = Solver(A,F,vecAlpha,guess,tol);

            case 0.4
                % Alpha Vectors
                vecAlphaA = linspace(deg2rad(90),deg2rad(100), 1e3);      % A = around zero
                vecAlphaB = linspace(deg2rad(100),deg2rad(180), 150);     % B = rest of the region
                vecAlpha = [vecAlphaA, vecAlphaB];
                
                nAlphaA = length(vecAlphaA);
                nAlphaB = length(vecAlphaB);
               
                % Guess vectors
                guessA = -1.*linspace(0,5,nAlphaA);          
                guessB = -1.*linspace(15,20,nAlphaB);

                % Axis limits for visualization
                AxLimX = [-6 12]; 
                AxLimY = [-10 10];

                % Solution
                [~, Re1A, Im1A, Re2A, Im2A] = Solver(A,F,vecAlphaA,guessA,tol);
                [~, Re1B, Im1B, Re2B, Im2B] = Solver(A,F,vecAlphaB,guessB,tol);

                Re1 = [Re1A, Re1B];
                Im1 = [Im1A, Im1B];
                Re2 = [Re2A, Re2B];
                Im2 = [Im2A, Im2B];

                % Remove spurious points
                Re1(abs(Im1) < 1e-12) = NaN;
                Im1(abs(Im1) < 1e-12) = NaN;
                Re2(abs(Im2) < 1e-12) = NaN;
                Im2(abs(Im2) < 1e-12) = NaN;

                % Rewrite vectors for plot
                Re1 = [0, Re1(~isnan(Re1))];
                Im1 = [0, Im1(~isnan(Im1))];
                Re2 = [0, Re2(~isnan(Re2))];
                Im2 = [0, Im2(~isnan(Im2))];  

            case 0.6
                % Alpha Vectors
                vecAlphaA = linspace(deg2rad(90),deg2rad(100),1e3);
                vecAlphaB = linspace(deg2rad(100),pi,150);
                
                nAlphaA = length(vecAlphaA);
                nAlphaB = length(vecAlphaB);
                
                % Guesses vector
                guessA = linspace(0,10,nAlphaA);
                guessB = linspace(10,20,nAlphaB);
                
                % Axis limits for visualization
                AxLimX = [-3 15]; 
                AxLimY = [-10 10];

                [~, Re1A, Im1A, Re2A, Im2A] = Solver(A,F,vecAlphaA,guessA,tol);
                [~, Re1B, Im1B, Re2B, Im2B] = Solver(A,F,vecAlphaB,guessB,tol);

                % Rewrite vectors for plot
                Re1 = [Re1A, Re1B];
                Im1 = [Im1A, Im1B];
                Re2 = [Re2A, Re2B];
                Im2 = [Im2A, Im2B];
                
            case 0.8
                % Alphas vector
                nAlpha = 100; 
                vecAlpha = linspace(0,pi,nAlpha);

                % Guess vector
                guess = 1.5.*linspace(0, pi, nAlpha);

                % Axis limit for visualization
                AxLimX = [-6 6];
                AxLimY = [-6 6];

                %   h solution
                [~, Re1, Im1, Re2, Im2] = Solver(A,F,vecAlpha,guess,tol);

            otherwise
                disp('Invalid Theta')
                return
        end

        % Plot
        figure
        % Define the limits for the axes
        xlim(AxLimX)
        ylim(AxLimY)
        xLimits = xlim;  % Get the current x-axis limits
        yLimits = ylim;  % Get the current y-axis limits
        
        if theta > 0.5
            fill(Re1, Im1, [1, 0.647, 0], 'FaceAlpha', 0.25, 'EdgeColor', 'none'); 
            hold on
            fill(Re2, Im2, [1, 0.647, 0], 'FaceAlpha', 0.25, 'EdgeColor', 'none');
            plot(Re1, Im1, 'Color',[1, 0.647, 0], 'LineWidth', 2);
            plot(Re2, Im2, 'Color' ,[1, 0.647, 0], 'LineWidth', 2);
            xline(0, 'Color', [0.3 0.3 0.3], 'LineWidth', 0.75); % x-axis
            yline(0, 'Color', [0.3 0.3 0.3], 'LineWidth', 0.75); % y-axis
            grid on
            title(['BI2 stability regions, $\theta = \;$', num2str(theta)])
            xlabel('$Re \{ h \lambda \}$')
            ylabel('$Im \{ h \lambda \}$')
            legend('Stable')
            hold off
        else
            fill([xLimits(1), xLimits(1), xLimits(2), xLimits(2)], [yLimits(1), yLimits(2), yLimits(2), yLimits(1)], [0.7 0.9 1], 'FaceAlpha', 0.5, 'EdgeColor', 'none');  % Background color
            hold on
            fill(Re1, Im1, [1 1 1], 'FaceAlpha', 1, 'EdgeColor', 'none');
            fill(Re2, Im2, [1 1 1], 'FaceAlpha', 1, 'EdgeColor', 'none');
            plot(Re1, Im1, 'Color',[0, 0.5, 1], 'LineWidth', 2);
            plot(Re2, Im2, 'Color',[0, 0.5, 1], 'LineWidth', 2);
            xline(0, 'Color', [0.3 0.3 0.3], 'LineWidth', 0.75); % x-axis
            yline(0, 'Color', [0.3 0.3 0.3], 'LineWidth', 0.75); % y-axis
            grid on
            title(['BI2 stability region, $\theta = \;$', num2str(theta)])
            xlabel('$Re \{ h \lambda \}$')
            ylabel('$Im \{ h \lambda \}$')
            legend('Stable')
            hold off
        end

    case 'IEX4'
        %   IEX4 operator
        F = @(h,a) -1/6*(I - A(a)*h)^-1 + 4*(I - A(a)*h/2)^-2 -27/2*(I - A(a)*h/3)^-3 + 32/3*(I - A(a)*h/4)^-4;

        %   Alpha vector
        nAlpha = 100; 
        vecAlpha = linspace(0, pi, nAlpha);

        %   Guess vector
        guess = -3.*linspace(0, pi, nAlpha); 

        %   h*lambda solution
        [~, Re1, Im1, Re2, Im2] = Solver(A,F,vecAlpha,guess,tol);


        % Plot
        % Define the limits for the axes
        xlim([-3 15])
        ylim([-15 15])
        xLimits = xlim;  % Get the current x-axis limits
        yLimits = ylim;  % Get the current y-axis limits
            
        fill([xLimits(1), xLimits(1), xLimits(2), xLimits(2)], [yLimits(1), yLimits(2), yLimits(2), yLimits(1)],[0.7 0.9 1], 'FaceAlpha', 0.5, 'EdgeColor', 'none');  % Background color
        hold on
        fill(Re1, Im1, [1 1 1], 'FaceAlpha', 1, 'EdgeColor', 'none');
        fill(Re2, Im2, [1 1 1], 'FaceAlpha', 1, 'EdgeColor', 'none');
        plot(Re1, Im1, 'Color',[0, 0.5, 1], 'LineWidth', 2);
        plot(Re2, Im2, 'Color' ,[0, 0.5, 1], 'LineWidth', 2);
        xline(0, 'Color', [0.3 0.3 0.3], 'LineWidth', 0.75); % x-axis
        yline(0, 'Color', [0.3 0.3 0.3], 'LineWidth', 0.75); % y-axis
        grid on
        title('IEX4 stability region')
        xlabel('$Re \{ h \lambda \}$')
        ylabel('$Im \{ h \lambda \}$')
        legend('Stable')

    otherwise
        disp('Invalid Method')
        return
end

end

function [h, Re1, Im1, Re2, Im2] = Solver(A,F,vecAlpha,h0,tol)

% [hVec, Re1, Im1, Re2, Im2] = Solver(A,F,vecAlpha,h0,tol)
%
% This function solves the problem "Find h ≥ 0 s.t. max(|eig(F (h, α))|) = 1" 
% and computes the h*lambda vectors
%
% INPUT:
% A             Matrix of your system
% F             Operator as function
% vecAlpha      Vector with the values of alpha
% n             Number of points in vecAlpha
% h0            Initial guess
% tol           Tolerances   
%
% OUTPUT:
% Re1           Real value of the first eigenvalue
% Im1           Immaginary value of the first eigenvalue
% Re2           Real value of the seconf eigenvalue
% Im2           Immaginary value of the second eigenvalue
%

        % fzero options
        opt = optimset('Display', 'off','TolFun', tol, 'TolX', tol); 

        % Initialization
        n = max(size(vecAlpha));
        h = zeros(1,n);
        lambda = zeros(2,n);
       
        for i = 1:n
            Prob = @(h) max(abs(eig(F(h,vecAlpha(i))))) - 1;
            h(i) = fzero(Prob, h0(i), opt);
            lambda(:,i) = eigs(A(vecAlpha(i)));
        end
        
        % Generate boundaries
        hlam1 = h .* lambda(1, :);
        hlam2 = h .* lambda(2, :);
        
        Re1 = real(hlam1);
        Im1 = imag(hlam1);
        Re2 = real(hlam2);
        Im2 = imag(hlam2);
        
end

function [x, t] = RK4Method(f, h, x0, tStart, tEnd)
% This function perfom the fourth order Runge - Kutta method
% 
% INPUT
% f(t,x):   function to solve
% h:        step size
% x0:       initial guess
% tStart:   initial time
% tEnd:     final time
% 
% OUTPUT
% x:        ODE solution
% t:        time vector 
% 
%--------------------------------------------------------------------------
    t       = tStart:h:tEnd;
    tSize   = max(size(t));
    sysSize = max(size(x0));
    x       = zeros(sysSize, tSize);
    x(:,1)  = x0;
    i       = 1;
    while i < tSize
        %   Predictor steps
        xP_new      = x(:,i) + h/2 * f(t(i), x(:,i));
        xP_new_2    = x(:,i) + h/2 * f(t(i) + h/2, xP_new);
        xP_new_3    = x(:,i) + h * f(t(i) + h/2, xP_new_2);
        %   Corrector step
        x(:,i + 1)  = x(:,i) + h*(1/6 * f(t(i), x(:,i)) + 1/3 * f(t(i) + h/2, xP_new) + 1/3 * f(t(i) + h/2 ,xP_new_2) + 1/6 * f(t(i + 1), xP_new_3) );

        i = i + 1;
    end
end