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

% Define thermal control variables
control.kp = 0.0002;                     % Control variable of the radiator
control.thetaMin = -0.4*pi;
control.thetaMax = 0;
control.R = 0.1;                    % Resistance [Ohm]
control.L = 0.001;                  % Inductance [Henry]
control.km = 0.3;                   % Motor Constant [Nm/A]
control.mRad = 0.2;                 % Radiator Mass [kg]


% Define thermal parameters of the system
parameters.Psun = 1350;                % Sun power [W/m^2]
parameters.C1 = 1.5e5;                 % Heat Capacity 1 [J/K]
parameters.C2 = 1187.5;                % Heat Capacity 2 [J/K]
parameters.C3 = parameters.C2;         % Heat Capacity 3 [J/K]
parameters.C4 = 30;                    % Heat Capacity 4 [J/K]
parameters.C5 = parameters.C4;         % Heat Capacity 5 [J/K]
parameters.G12 = 10;                   % Thermal Conductance 1-2 [W/K]
parameters.G13 = 10;                   % Thermal Conductance 1-3 [W/K]
parameters.G14 = 10;                   % Thermal Conductance 1-4 [W/K]
parameters.G15 = 10;                   % Thermal Conductance 1-5 [W/K]
parameters.alpha1 = 0.6;               % Absorptivity 1 [-]
parameters.alpha2 = 0.78;              % Absorptivity 2 [-]
parameters.alpha3 = parameters.alpha2; % Absorptivity 3 [-]
parameters.eps1 = 0.45;                % Emissivity 1 [-]
parameters.eps2 = 0.75;                % Emissivity 2 [-]
parameters.eps3 = parameters.eps2;     % Emissivity 3 [-]
parameters.epsMin = 0.01;              % Emissivity Minmum [-]
parameters.epsMax = 0.98;              % Emissivity Maximum [-]
parameters.Tds = 3;                    % Temperature Deep Space [K]
parameters.T0 = 298.15;                % Initial Temperature for each Node [K]
parameters.Tref = 294.15;              % Reference Temperature [K]

% Define Spacecraft Geometry
geometry.width = 0.5;                                       % Spacecraft Width [m]                              
geometry.thSA = 0.01;                                       % Solar Array Thickness [m]      
geometry.lSA = 0.95;                                        % Solar Array Length [m] 
geometry.lRad = 0.5;                                        % Radiator Length [m]
geometry.lBody = 0.5;                                       % Main Body Length [m]
geometry.hBody = 1.5;                                       % Main Body Height [m]

geometry.Jr = 1/3*control.mRad*geometry.lRad^2;             % Moment of inertia of the radiator [kg*m^2]
geometry.AsunBody = geometry.width * geometry.lBody;        % Sun-Side Area Body [m^2]
geometry.AsunSA = geometry.width * geometry.lSA;            % Sun-Side Area Solar Array [m^2]
geometry.AtotBody = geometry.AsunBody + 2*geometry.lBody*geometry.hBody + 2*geometry.width*(geometry.hBody - geometry.thSA); % Total Body Area [m^2]
geometry.AtotSA = geometry.AsunSA + 2*geometry.lSA*geometry.thSA + geometry.width*geometry.thSA;                             % Total Solar Array Area [m^2]
geometry.AtotRad = 2*geometry.width*geometry.lRad;          % Radiator Area [m^2]

tComp = zeros(1,10);

for i = 1:10
% Perform Propagation
tspan = [0 50*3600]; % Time span (50 hours, converted to seconds)

x0 = [parameters.T0, parameters.T0, parameters.T0, parameters.T0, parameters.T0, 0, control.thetaMin, 0]; % Initial state
options = odeset('RelTol',1e-12,'AbsTol',1e-12);

tic
[t, xsol] = ode15s(@(t, xsol) RHS1(xsol, parameters, control, geometry), tspan, x0, options);
tComp(i) = toc;
end

Time = mean(tComp);

% Plot temperatures of the first node
Tref = parameters.Tref;

figure;
hold on;
plot(t/3600, xsol(:, 1), 'Color',[0 0.4470 0.7410]); % Temperature at node 1
plot(t/3600, Tref * ones(size(t)),'Color', [0.8500 0.3250 0.0980],'LineStyle','--'); % Reference temperature
xline(36000 / 3600, 'Color',[0.9290 0.6940 0.1250], 'Linestyle', '--','LineWidth',1.1); % 10 hour mark
plot(t/3600, (Tref + 0.001 * Tref) * ones(size(t)), 'k-.'); % Max temperature
plot(t/3600, (Tref - 0.001 * Tref) * ones(size(t)), 'k-.'); % Min temperature
xlabel('Time [h]');
ylabel('Temperature [K]');
title('Temperature Evolution of Main Body');
legend('$T_1$','$T_{ref}$','t = 10 h', 'Oscillation Limits')
legend('Location', 'best');
grid on;
hold off;

% Plot temperatures of all nodes
figure;
hold on;
plot(t/3600, xsol(:, 1), 'Color',[0 0.4470 0.7410]); % Temperature at node 1
plot(t/3600, xsol(:, 2),'Color',[0.8500 0.3250 0.0980] ); % Temperature at node 2
plot(t/3600, xsol(:, 3),'Color',[0.9290 0.6940 0.1250]); % Temperature at node 3
plot(t/3600, xsol(:, 4),'Color',[0.4660 0.6740 0.1880]); % Temperature at node 4
plot(t/3600, xsol(:, 5),'Color',[0.4940 0.1840 0.5560]); % Temperature at node 5
ylim([275 325])
xlabel('Time [h]');
ylabel('Temperature [K]');
title('Temperature Evolution of Thermal Nodes');
legend('$T_1$','$T_2$','$T_3$', '$T_4$','$T_5$')
legend('Location', 'best');
grid on;
hold off;

% Plot the radiator angle
figure;
hold on;
plot(t/3600, xsol(:, 7), 'Color',[0 0.4470 0.7410]); % Radiator angle
plot(t/3600, 0 * ones(size(t)), 'k-.'); % Max angle
plot(t/3600, (-0.4*pi) * ones(size(t)), 'k-.'); % Min angle
xlabel('Time [h]');
ylabel('Radiator Angle [rad]');
title('Evolution of the Radiator Angle');
legend('$\theta$', 'Limits')
legend('Location', 'best');
grid on;
hold off;

% Plot the rediator emissivity
eps = parameters.epsMin + (parameters.epsMax - parameters.epsMin) ./ (control.thetaMax - control.thetaMin) .* (xsol(:, 7) - control.thetaMin);

figure;
hold on;
plot(t/3600, eps, 'Color',[0 0.4470 0.7410]); % Radiator emissivity
plot(t/3600, parameters.epsMax * ones(size(t)), 'k-.'); % Max emissivity
plot(t/3600, parameters.epsMin * ones(size(t)), 'k-.'); % Min emissivity
xlabel('Time [h]');
ylabel('Radiator Emissivity [-]');
title('Evolution of the Radiator Emissivity');
legend('$\epsilon$', 'Limits')
legend('Location', 'best');
grid on;
hold off;

% Plot the DC current
figure;
plot(t/3600, xsol(:, 6), 'Color',[0 0.4470 0.7410]); % Current
xlabel('Time [h]');
ylabel('Current [A]');
title('Evolution of the DC Motor Current i');
grid on;

% Subplots of the DC Current
figure;
subplot(3, 1, 3); 
plot(t/3600, xsol(:, 6), 'Color',[0 0.4470 0.7410]); % Current
xlabel('t [h]');
ylabel('i [A]');
xlim([1.35272 1.3528])
ylim([-2e-4 13e-4])
title('Current Zoom 2');
grid on;

subplot(3, 1, 1); 
plot(t(1:8500)/3600, xsol(1:8500, 6), 'Color',[0 0.4470 0.7410]); % Current
xlabel('t [h]');
ylabel('i [A]');
ylim([-10e-8 5e-8])
xlim([0 3.5])
title('Current Zoom 1');
grid on;

subplot(3, 1, 2); 
plot(t/3600, xsol(:, 6), 'Color',[0 0.4470 0.7410]); % Current
xlabel('t [h]');
ylabel('i [A]');
ylim([-5e-3 1e-3])
xlim([0.143745 0.14379])
title('Current Zoom 3');
grid on;



% Plot the Voltage
Vin = zeros(size(t));
for i = 1:size(t)
    Vin(i) = control.kp*(xsol(i, 1)-parameters.Tref);
    if xsol(i, 7) > control.thetaMax && xsol(i, 1) > parameters.Tref
        Vin(i) = 0;
    elseif xsol(i, 7) < control.thetaMin && xsol(i, 1) < parameters.Tref
        Vin(i) = 0;
    end
end

figure
plot(t/3600, Vin,'Color',[0 0.4470 0.7410]); % Voltage
xlabel('Time [h]');
ylabel('Voltage [V]');
title('Evolution of the Input Voltage $V_{in}$');
grid on;

% Plot the Angular Velocity
figure;
plot(t/3600, xsol(:, 8), 'Color',[0 0.4470 0.7410]); % Current
xlabel('Time [h]');
ylabel('Angular Velocity [rad/s]');
title('Evolution of the Angular Velocity $\dot{\theta}$');
grid on;


%% EXECRISE 2

clc
clear
close all


% Define initial parameters
accelerometer.Msc = 300;                        % Spacecraft Mass [kg]
accelerometer.Ma = 0.32;                        % Seismic Mass [kg]
accelerometer.Kacc = 1;                         % Proportional Coefficient [Vs/m]

valve.Mv = 0.1;                                 % Spool Mass [kg]
valve.kv = 1e3;                                 % Valve Spring [N/m]
valve.bv = 1e3;                                 % Valve Damper [Ns/m]
valve.alpha = 2.1e-2;                           % Solenoid Constant [1/H]
valve.beta = -60;                               % Solenoid Gain [1/Hm]
valve.Amin = 4.7e-12;                           % Minimum Area [m^2]
valve.xvMax = 1e-5;                             % Maximum Extension [m]

thruster.k = 1.66;                              % Heat Ratio [-]
thruster.Ptank = 2e5;                           % Tank Pressure [Pa]
thruster.Ttank = 240;                           % Tank Temperature [K]
thruster.R = 63.32754;                          % Gas Constant [J/kg*K]
thruster.q = 1.6e-19;                           % Charge [C]
thruster.dV = 2000;                             % Voltage [V]
thruster.Mi = 2.188e-25;                        % Ion Mass [kg]

drag.omegaS = 1.658226e-6;                      % Secular Pulsation [rad/s]
drag.omegaO = 1.160758e-3;                      % Orbital Pulsation [rad/s]

% Define bounds for parameters
lb = [1.5e3, 5e-5, 0.1, 1e4]; % Lower bounds
ub = [2e4, 3e-3, 10, 8e4];    % Upper bounds

% Run Genetic Algorithm
rng(01012025)
optionsga = optimoptions('ga','PopulationSize',50,'MaxGenerations',150,'Display','iter','UseParallel',false);
% ACTIVATE PARALLEL COMPUTING FOR ENHANCED PERFORMANCES
[bestOpt,~] = ga(@(opt) costFunction(opt, accelerometer, valve, thruster, drag),4,[],[],[],[],lb,ub,[],optionsga);

% Update Optimized Parameters
accelerometer.ba = bestOpt(1);
accelerometer.ka = bestOpt(2);
amplifier.Rin = bestOpt(3);
amplifier.Rf = bestOpt(4);

% Perform Propagation
for i = 1:10
Period = 3*2*pi/drag.omegaO;

x0 = [0, 0, valve.xvMax, 0, 0]; % Initial state
options = odeset('RelTol',1e-12,'AbsTol',1e-12);

tic
[t, xsol] = ode15s(@(t, xsol) RHS2(t,xsol, accelerometer, amplifier, valve,thruster,drag), [0 Period], x0, options);
tMat = toc;
end

TimeMat = mean(tMat);


% Compute Drag and Thrust
D = 2.2 - cos(drag.omegaS*t) + 1.2*sin(drag.omegaO*t).*cos(drag.omegaO*t);

rhoXe = thruster.Ptank / (thruster.R * thruster.Ttank);
Av = valve.Amin + valve.xvMax * (valve.xvMax - xsol(:,3));
mdot = Av .* sqrt(thruster.k * rhoXe * thruster.Ptank * (2 / (thruster.k + 1))^((thruster.k + 1) / (thruster.k - 1)));
T = mdot .* sqrt(2 * thruster.q * thruster.dV / thruster.Mi);

% Compute the mean error between thrust and drag
id = find(t>30,1);
MatErr = mean(abs(T(id:end)-D(id:end)/1000));

% Plot the Drag and Thrust
figure;
hold on;
plot(t/3600, T.*1000, 'DisplayName', 'Thrust'); 
plot(t/3600, D, '--', 'DisplayName', 'Drag','Color',[0.8500, 0.3250, 0.0980]); 
xlabel('Time [h]');
xlim([0 3*To/3600])
ylabel('Force [mN]');
title('Thrust and Drag evolution in time');
legend('Location', 'best');
grid on;
hold off;

% Drag and Thrust zoom plot
figure;
hold on;
plot(t, T.*1000, 'DisplayName', 'Thrust'); 
plot(t, D, '--', 'DisplayName', 'Drag','Color',[0.8500, 0.3250, 0.0980]); 
xlim([0 30])
xlabel('Time [s]');
ylabel('Force [mN]');
title('Thrust and Drag Zoom');
legend('Location', 'best');
grid on;
hold off;

% Accelerometer displacement plot
figure;
plot(t/3600, xsol(:,1)); 
xlim([0 3*To/3600])
xlabel('Time [h]');
ylabel('Displacement [m]');
title('Accelerometer Mass Displacement');
grid on;

% Accelerometer speed subplots
figure;
subplot(2,1,1)
plot(t/3600, xsol(:,2)); 
xlim([0 3*To/3600])
xlabel('Time [h]');
ylabel('Speed [m/s]');
title('Accelerometer Mass Speed');
grid on;

subplot(2,1,2)
plot(t, xsol(:,2)); 
xlim([0 30])
xlabel('Time [s]');
ylabel('Speed [m/s]');
title('Accelerometer Speed Zoom');
grid on;

% Valve displacement
figure;
subplot(2,1,1)
plot(t/3600, xsol(:,3)); 
xlim([0 3*To/3600])
xlabel('Time [h]');
ylabel('Displacement [m]');
title('Valve Displacement');
grid on;

subplot(2,1,2)
plot(t, xsol(:,3)); 
xlim([0 30])
xlabel('Time [s]');
ylabel('Displacement [m]');
title('Valve Displacement Zoom');
grid on;

% Valve speed 
figure;
subplot(2,1,1)
plot(t/3600, xsol(:,4)); 
xlim([0 3*To/3600])
xlabel('Time [h]');
ylabel('Speed [m/s]');
title('Valve Speed');
grid on;

subplot(2,1,2)
plot(t, xsol(:,4)); 
xlim([0 30])
xlabel('Time [s]');
ylabel('Speed [m/s]');
title('Valve Speed Zoom');
grid on;

% Valve current 
figure;
subplot(2,1,1)
plot(t/3600, xsol(:,5)); 
xlim([0 3*To/3600])
xlabel('Time [h]');
ylabel('Current [A]');
title('Valve Current');
grid on;

subplot(2,1,2)
plot(t, xsol(:,5)); 
xlim([0 30])
xlabel('Time [s]');
ylabel('Current [A]');
title('Valve Current Zoom');
grid on;

% Valve Area 
figure;
hold on
plot(t/3600, Av); 
yline(4.7e-12,'LineStyle','--','Color', [0.3, 0.3, 0.3],LineWidth=2)
yline(4.7e-12+valve.xvMax^2,'LineStyle','--','Color', [0.3, 0.3, 0.3],LineWidth=2)
xlim([0 3*To/3600])
xlabel('Time [h]');
ylabel('Area [$m^2$]');
legend('$A_v$', 'Limits', 'Location','northeast')
title('Valve Area');
grid on;

%% Simscape Exercise

% Run the simulation
for i = 1:10
Period = 3*2*pi/drag.omegaO;

tic
sim("Assignment2_2.slx");
tSim = toc;
end

TimeSim = mean(tSim);

% Retrieve variables from the simulation
T = out.T;
D = out.D;
t = out.t;
xv = out.xv;
vv = out.vv;
xa = out.xa;
va = out.va;
i = out.i;
Av = out.Av;

% Compute errors between thrust and drag
idSim = find(t>30,1); 
SimErr = mean(abs(T(idSim:end)-D(idSim:end)));

% Plot the Drag and Thrust
figure;
hold on;
plot(t/3600, T.*1000, 'DisplayName', 'Thrust'); 
plot(t/3600, D.*1000, '--', 'DisplayName', 'Drag','Color',[0.8500, 0.3250, 0.0980]); 
xlabel('Time [h]');
xlim([0 3*To/3600])
ylabel('Force [mN]');
title('Thrust and Drag evolution in time');
legend('Location', 'best');
grid on;
hold off;

% Drag and Thrust zoom plot
figure;
hold on;
plot(t, T.*1000, 'DisplayName', 'Thrust'); 
plot(t, D.*1000, '--', 'DisplayName', 'Drag','Color',[0.8500, 0.3250, 0.0980]); 
xlim([0 30])
xlabel('Time [s]');
ylabel('Force [mN]');
title('Thrust and Drag Zoom');
legend('Location', 'best');
grid on;
hold off;

% Accelerometer displacement plot
figure;
plot(t/3600, xa); 
xlim([0 Period/3600])
xlabel('Time [h]');
ylabel('Displacement [m]');
title('Accelerometer Mass Displacement');
grid on;

% Accelerometer speed subplots
figure;
subplot(2,1,1)
plot(t/3600, va); 
xlim([0 Period/3600])
xlabel('Time [h]');
ylabel('Speed [m/s]');
title('Accelerometer Mass Speed');
grid on;

subplot(2,1,2)
plot(t, va); 
xlim([0 30])
xlabel('Time [s]');
ylabel('Speed [m/s]');
title('Accelerometer Speed Zoom');
grid on;

% Valve displacement
figure;
subplot(2,1,1)
plot(t/3600,xv); 
xlim([0 Period/3600])
xlabel('Time [h]');
ylabel('Displacement [m]');
title('Valve Displacement');
grid on;

subplot(2,1,2)
plot(t, xv); 
xlim([0 30])
xlabel('Time [s]');
ylabel('Displacement [m]');
title('Valve Displacement Zoom');
grid on;

% Valve speed 
figure;
subplot(2,1,1)
plot(t/3600, -vv); 
xlim([0 Period/3600])
xlabel('Time [h]');
ylabel('Speed [m/s]');
title('Valve Speed');
grid on;

subplot(2,1,2)
plot(t, -vv); 
xlim([0 30])
xlabel('Time [s]');
ylabel('Speed [m/s]');
title('Valve Speed Zoom');
grid on;

% Valve current 
figure;
subplot(2,1,1)
plot(t/3600, i); 
xlim([0 Period/3600])
xlabel('Time [h]');
ylabel('Current [A]');
title('Valve Current');
grid on;

subplot(2,1,2)
plot(t, i); 
xlim([0 30])
xlabel('Time [s]');
ylabel('Current [A]');
title('Valve Current Zoom');
grid on;

% Valve Area 
figure;
hold on
plot(t/3600, Av); 
yline(4.7e-12,'LineStyle','--','Color', [0.3, 0.3, 0.3],LineWidth=2)
yline(4.7e-12+valve.xvMax^2,'LineStyle','--','Color', [0.3, 0.3, 0.3],LineWidth=2)
xlim([0 Period/3600])
xlabel('Time [h]');
ylabel('Area [$m^2$]');
legend('$A_v$', 'Limits', 'Location','northeast')
title('Valve Area');
grid on;

%% FUNCTION

function dxdt = RHS1(xsol,parameters,control,geometry)

% dxdt = RHS1(xsol,parameters,control)
%
% Returns the right-hand side of the problem, given initial parameters and
% constants
%
% INPUT:
% xsol              Initial solution at each timestep
% parameters        Parameters of the spacecraft
% control           Parameters of the thermal control
% geometry          Geometry of the spacecraft
%
% OUTPUT:
% dxdt              Right-hand side of the problem

    % Constants
    sigma = 5.670374419e-8; % Stefan-Boltzmann constant [W/m^2K^4]
    
    % Retrieve state variables
    T1 = xsol(1); 
    T2 = xsol(2); 
    T3 = xsol(3);
    T4 = xsol(4); 
    T5 = xsol(5);
    i = xsol(6);
    theta = xsol(7); 
    dTheta = xsol(8);

    % Compute control voltage
    Vin = control.kp*(T1 - parameters.Tref);

    % Ensure Radiator Angle is bounded 
    if theta > control.thetaMax && T1 > parameters.Tref
        Vin = 0;
    elseif theta < control.thetaMin && T1 < parameters.Tref
        Vin = 0;
    end

    
    % Radiator emissivity as a function of theta
    epsRad = parameters.epsMin + (parameters.epsMax - parameters.epsMin) / (control.thetaMax - control.thetaMin) * (theta - control.thetaMin);
    
    % Compute derivatives
    dT1 = (parameters.Psun*geometry.AsunBody*parameters.alpha1 + parameters.G12*(T2-T1) + parameters.G13*(T3-T1) + parameters.G14*(T4-T1) + parameters.G15*(T5-T1) - parameters.eps1*sigma*geometry.AtotBody*(T1^4-parameters.Tds^4))/parameters.C1;

    dT2 = (parameters.Psun*geometry.AsunSA*parameters.alpha2 + parameters.G12*(T1-T2) - parameters.eps2*sigma*geometry.AtotSA*(T2^4-parameters.Tds^4))/parameters.C2;

    dT3 = (parameters.Psun*geometry.AsunSA*parameters.alpha3 + parameters.G13*(T1-T3) - parameters.eps3*sigma*geometry.AtotSA*(T3^4-parameters.Tds^4))/parameters.C3;

    dT4 = (parameters.G14*(T1-T4) - epsRad*sigma*geometry.AtotRad*(T4^4-parameters.Tds^4))/parameters.C4;

    dT5 = (parameters.G15*(T1-T5) - epsRad*sigma*geometry.AtotRad*(T5^4-parameters.Tds^4))/parameters.C5;

    % Current derivative
    di = Vin/control.L - control.km*dTheta/control.L - control.R/control.L * i;

    % Radiator dynamics 
    d2Theta = control.km *i / geometry.Jr;

    % Output derivatives
    dxdt = [dT1; dT2; dT3; dT4; dT5; di; dTheta; d2Theta];

end

function dxdt = RHS2(t,xsol,accelerometer,amplifier,valve,thruster,drag)

% dxdt = RHS2(xsol,accelerometer,amplifier,valve,thruster,drag)
%
% Returns the right-hand side of the problem, given initial parameters and
% constants
%
% INPUT:
% t                 Time
% xsol              Initial solution at each timestep
% accelerometer     Parameters of the accelerometer
% amplifier         Parameters of the amplifier
% valve             Parameters of the valve
% thruster          Parameters of the thruster
% drag              Parameters of the drag
%
% OUTPUT:
% dxdt              Right-hand side of the problem

    
    % Retrieve state variables
    xa = xsol(1);
    va = xsol(2);
    xv = xsol(3);
    vv = xsol(4);
    i = xsol(5);

    if xv < 0
        xv= 0;
    elseif xv > valve.xvMax
        xv= valve.xvMax;
    end

    
    % Compute other required parameters
    
    D = (2.2 - cos(drag.omegaS*t) + 1.2*sin(drag.omegaO*t)*cos(drag.omegaO*t))/1000;       % Drag [N]

    Av = valve.Amin + valve.xvMax*(valve.xvMax - xv);       % Area of the orifice [m^2]

    rhoXe = thruster.Ptank/(thruster.R*thruster.Ttank);     % Xenon denisty [kg/m^3]

    mdot = Av*sqrt(thruster.k*rhoXe*thruster.Ptank*(2/(thruster.k+1))^((thruster.k+1)/(thruster.k-1)));     % Xenon mass flow [kg/s]

    T = mdot*sqrt(2*thruster.q*thruster.dV/thruster.Mi);        % Thrust [N]

    L = 1/(valve.alpha + valve.beta*xv);

    dL = -valve.beta/(valve.alpha + valve.beta*xv)^2;

    Vout = va*accelerometer.Kacc;

    Vhat = -(amplifier.Rf/amplifier.Rin)*Vout;

    % Compute derivatives
    dXa = va;

    dVa = (T-D)/accelerometer.Msc - accelerometer.ba/accelerometer.Ma*va - accelerometer.ka/accelerometer.Ma*xa;

    dXv = vv;

    dVv = 1/2*i^2*dL/valve.Mv - valve.bv/valve.Mv*vv - valve.kv/valve.Mv*xv;

    dI = 1/L * Vhat;

    % Output derivatives
    dxdt = [dXa; dVa; dXv; dVv; dI];

end

function cost = costFunction(opt,accelerometer,valve,thruster,drag)

% cost = costFunction(opt,accelerometer,valve,thruster,drag)
%
% Returns the cost function defined for this problem
%
% INPUT:
% opt               Parameters to be optimized
% accelerometer     Parameters of the accelerometer
% valve             Parameters of the valve
% thruster          Parameters of the thruster
% drag              Parameters of the drag
%
% OUTPUT:
% cost              Cost function

    % Retrieve parameters
    ba = opt(1);
    ka = opt(2);
    Rin = opt(3);
    Rf = opt(4);
    
    % Update parameters
    accelerometer.ba = ba;
    accelerometer.ka = ka;
    amplifier.Rin = Rin;
    amplifier.Rf = Rf;
    
    % Solve ODE
    x0 = [0, 0, valve.xvMax, 0, 0];
    options = odeset('RelTol', 1e-6, 'AbsTol', 1e-6);
    [t, xsol] = ode15s(@(t, xsol) RHS2(t, xsol, accelerometer, amplifier, valve, thruster, drag), 0:1:3*2*pi/drag.omegaO, x0, options);

    % Compute thrust and drag
    D = (2.2 - cos(drag.omegaS*t) + 1.2*sin(drag.omegaO*t).*cos(drag.omegaO*t))/1000;

    rhoXe = thruster.Ptank / (thruster.R * thruster.Ttank);
    Av = valve.Amin + valve.xvMax * (valve.xvMax - xsol(:,3));
    mdot = Av .* sqrt(thruster.k * rhoXe * thruster.Ptank * (2 / (thruster.k + 1))^((thruster.k + 1) / (thruster.k - 1)));
    T = mdot .* sqrt(2 * thruster.q * thruster.dV / thruster.Mi);
    
    % Compute cost function
    cost = trapz(t, abs(T - D));
end


    

