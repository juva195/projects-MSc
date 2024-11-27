
clc
clear
close all
InitialSettings

%% INITIAL VALUES

a = 1.6882e+4; %[km]
e = 0.6026; 
i = 19.3243*pi/180; %[rad]

GTratio = 13/3;
k = 13;
m = 3;
parameters.CD = 2.1;
parameters.AMratio = 0.0569; %[m^2/kg]

raan = 277.5732*pi/180; %[rad]
omega = 162.3587*pi/180; %[rad]
f = 270*pi/180; %[rad]


E = f2E(f,e);
M = E2M(E,e);

parameters.mu = astroConstants(13);
parameters.Re = astroConstants(23);
parameters.J2 = astroConstants(9); % Second Zonal Armonic 
parameters.omegaE = 7.291597763887421e-05;
parameters.thetaG0 = 0;



period = 2*pi*sqrt(a^3/parameters.mu);
revolution = 60*(60*23+56);
periodsPerDay = 60*60*24/period;
days = 300/periodsPerDay; % Days to plot
orbit = [a,e,i,raan,omega,f];

%% INITIAL ORBIT

% ODE parameter
[r, v] = orbitalToCar(a,e,i,raan,omega,0,parameters.mu);
tspan = 0:dt:period;
y0 = [r;v];

% Propagate orbit
[~, yInitial] = ode113( @(t,y) ode2bp(t,y,parameters.mu), tspan, y0, options);
rInitial = yInitial(:,1:3);
vInitial = yInitial(:,4:6);

% Plot orbit
figure();
background("Stars")
hold on;
planet3D()
colormap("autumn")
patch([rInitial(:,1); nan], [rInitial(:,2); nan], [rInitial(:,3); nan], [vecnorm(vInitial,2,2); nan] ,'LineWidth',2,'FaceColor','none','EdgeColor','interp');
cb = colorbar('Color','w','FontSize',22,'position',[.83, .2, .015, .5],'TickLabelInterpreter','latex');
cb.Label.String = "Velocity $km/s$";
cb.Label.Interpreter = "latex";
clim([min(vecnorm(vInitial,2,2)), max(vecnorm(vInitial,2,2))])
legend("","Nominal orbit",'Color',"k",'TextColor',"w")
view(-93, 90);
axis tight equal

%% GROUND TRACK

groundTrack(orbit,1,10,default,default,default,default);

groundTrack(orbit,periodsPerDay,10,default,default,default,default);

[Alpha,Delta,Long,Lat] =  groundTrack(orbit,periodsPerDay*10,10,default,default,default,default);

%% REPEATING GROUND TRACK

aRGT = repeatingGroundTrack(m,k,default,default);

orbitRGT = [aRGT,e,i,raan,omega,f];

periodRGT = 2*pi*sqrt(aRGT^3/parameters.mu);

periodsPerDayRGT = 60*(60*23+56)/periodRGT;

[~,~,LongRGT,LatRGT] =  groundTrack(orbitRGT,13,10,default,default,default,default);

%% PERTURBED REPEATING GROUND TRACK

dRAAN = @(x) -cos(i)*3/2* (sqrt(parameters.mu)*parameters.J2*parameters.Re^2) / ((1-e^2)^2*x^(7/2));

dOmega = @(x) -3/2*(sqrt(parameters.mu)*parameters.J2*parameters.Re^2)/((1-e^2)^2*x^(7/2)) * (5/2*sin(i)^2-2) ;

mdotr = @(x) 3/2 * (sqrt(parameters.mu)*parameters.J2*parameters.Re^2)/((1-e^2)^(3/2)*x^(7/2))*(1-3/2*sin(i)^2);

n = @(x) sqrt(parameters.mu/(x^3));

eqn = @(x) (parameters.omegaE-dRAAN(x))/(n(x)+dOmega(x)+mdotr(x)) - m/k;

aPRGT = fzero(eqn, aRGT);

% Plot
orbitPRGT = [aPRGT,e,i,raan,omega,f];
groundTrack(orbitPRGT,13*5,10,default,default,default,default);
groundTrackTransparent(orbitRGT,13,10,default,default,default,default);

%% PERTURBATIONS USING CARTESIAN COORDINATES - 2 DAYS

tspan = 0:10:periodsPerDay*period*2;

% Set options for the ODE solver
options = odeset( 'RelTol', 1e-12, 'AbsTol', 1e-13);

[rcar, vcar] = orbitalToCar(a,e,i,raan,omega,f,parameters.mu);

car = [rcar; vcar];

% Perform the integration
[T, CAR] = ode113( @(t,y) eqMotionCar(t, y, @(t,y) PerAccCalculatorCar(t, y, parameters), parameters ), tspan, car, options);


%% GROUND TRACK PLOT

groundTrackUnpropagated(car,period,parameters,10,default,default,default,default);

groundTrackUnpropagated(car,period*periodsPerDay,parameters,10,default,default,default,default);

groundTrackUnpropagated(car,period*periodsPerDay*10,parameters,10,default,default,default,default);

%% CONVERSION IN KEPLERIAN ELEMENTS AND PLOTTING - 2 DAYS

kep = zeros(6,size(CAR,1));
kepMean = zeros(6,size(CAR,1));
days = 10;

for j = 1:size(CAR,1)

   [aTemp, eTemp, iTemp, raanTemp, omegaTemp, fTemp] = carToOrbital(CAR(j,1:3),CAR(j,4:6),default);
    
   kep(1,j)= aTemp;
   kep(2,j)= eTemp;
   kep(3,j)= iTemp;
   kep(4,j)= raanTemp;
   kep(5,j)= omegaTemp;
   kep(6,j)= fTemp;

end

c = polyfit(T,kep(1,:),1);
kepMean(1,:) = polyval(c,T);
c = polyfit(T,kep(2,:),1);
kepMean(2,:) = polyval(c,T);
c = polyfit(T,kep(3,:),1);
kepMean(3,:) = polyval(c,T);
c = polyfit(T,kep(4,:),1);
kepMean(4,:) = polyval(c,T);
c = polyfit(T,kep(5,:),1);
kepMean(5,:) = polyval(c,T);
c = polyfit(T,kep(6,:),1);
kepMean(6,:) = polyval(c,T);

figure 
plot(T/period,kep(1,:))
hold on
plot(T/period,kepMean(1,:),'Color','k','LineWidth',2)
title(sprintf('Evolution of the semimajor axis in %d periods', ceil(T(end)/period)))
ylabel("Semi-major axis $[km$]")
xlabel("Periods")
legend("Cartesian propagation","Filtered cartesian propagation")

figure 
plot(T/period,kep(2,:))
hold on
plot(T/period,kepMean(2,:),'Color','k','LineWidth',2)
title(sprintf('Evolution of the eccentricity in %d periods', ceil(T(end)/period)))
ylabel("Eccentricity")
xlabel("Periods")
legend("Cartesian propagation","Filtered cartesian propagation")

figure 
plot(T/period,kep(3,:)*180/pi)
hold on
plot(T/period,kepMean(3,:)*180/pi,'Color','k','LineWidth',2)
title(sprintf('Evolution of the inclination in %d periods', ceil(T(end)/period)))
ylabel("Inclination $[deg]$")
xlabel("Periods")
legend("Cartesian propagation","Filtered cartesian propagation")

figure 
plot(T/period,kep(4,:)*180/pi)
hold on
plot(T/period,kepMean(4,:)*180/pi,'Color','k','LineWidth',2)
title(sprintf('Evolution of the RAAN in %d periods', ceil(T(end)/period)))
ylabel("RAAN $[deg]$")
xlabel("Periods")
legend("Cartesian propagation","Filtered cartesian propagation")

figure 
plot(T/period,kep(5,:)*180/pi)
hold on
plot(T/period,kepMean(5,:)*180/pi,'Color','k','LineWidth',2)
title(sprintf('Evolution of the argument of pericentre in %d periods', ceil(T(end)/period)))
ylabel("Argument of pericentre $[deg]$")
xlabel("Periods")
legend("Cartesian propagation","Filtered cartesian propagation")

figure
plot(T/period,kep(6,:)*180/pi)
hold on
plot(T/period,kepMean(6,:)*180/pi,'Color','k','LineWidth',2)
title(sprintf('Evolution of the true anomaly in %d periods', ceil(T(end)/period)))
ylabel("True anomaly $[deg]$")
xlabel("Periods")
legend("Cartesian propagation","Filtered cartesian propagation")

%% PERTURBATIONS USING CARTESIAN COORDINATES - 300 PERIODS

tspan = 0:10:period*300;

% Set options for the ODE solver
options = odeset( 'RelTol', 1e-12, 'AbsTol', 1e-13);

[rcar, vcar] = orbitalToCar(a,e,i,raan,omega,f,parameters.mu);

car = [rcar; vcar];

% Perform the integration
[T, CAR] = ode113( @(t,y) eqMotionCar(t, y, @(t,y) PerAccCalculatorCar(t, y, parameters), parameters ), tspan, car, options);

%% PERTURBATIONS USING GAUSS EQUATIONS - 300 PERIODS

% Perform the integration
[~, ORB] = ode113( @(t,y) eqMotionKep(t, y, @(t,y) PerAccCalculatorKep(t, y, parameters), parameters ), tspan, orbit, options);


%% ORBIT PLOT - 300 PERIODS

r = CAR(:,1:3)';

figure
title('Evolution of the orbit');
background("Stars")
hold on;
planet3D()
colormap("autumn")
patch([r(1,:), nan], [r(2,:), nan], [r(3,:), nan], [T; nan] ,'LineWidth',2,'FaceColor','none','EdgeColor','interp');
cb = colorbar('Color','w','FontSize',22,'position',[.8, .25, .015, .5],'TickLabelInterpreter','latex');
cb.Label.String = "Time $s$";
cb.Label.Interpreter = "latex";
clim([T(1), T(end)])
view(100, 5)
axis tight equal

%% CONVERSION IN KEPLERIAN ELEMENTS AND PLOTTING - 300 PERIODS

kep = zeros(6,size(CAR,1));
kepMean = zeros(6,size(CAR,1));
days = 10;

for j = 1:size(CAR,1)

   [aTemp, eTemp, iTemp, raanTemp, omegaTemp, fTemp] = carToOrbital(CAR(j,1:3),CAR(j,4:6),default);
    
   kep(1,j)= aTemp;
   kep(2,j)= eTemp;
   kep(3,j)= iTemp;
   kep(4,j)= raanTemp;
   kep(5,j)= omegaTemp;
   kep(6,j)= fTemp;

end

% Filter cartesian
c = polyfit(T,kep(1,:),3);
kepMean(1,:) = polyval(c,T);
c = polyfit(T,kep(2,:),3);
kepMean(2,:) = polyval(c,T);
c = polyfit(T,kep(3,:),3);
kepMean(3,:) = polyval(c,T);
c = polyfit(T,kep(4,:),3);
kepMean(4,:) = polyval(c,T);
c = polyfit(T,kep(5,:),3);
kepMean(5,:) = polyval(c,T);
c = polyfit(T,kep(6,:),3);
kepMean(6,:) = polyval(c,T);

% Filter gauss
c = polyfit(T,ORB(:,1),3);
gaussMean(1,:) = polyval(c,T);
c = polyfit(T,ORB(:,2),3);
gaussMean(2,:) = polyval(c,T);
c = polyfit(T,ORB(:,3),3);
gaussMean(3,:) = polyval(c,T);
c = polyfit(T,ORB(:,4),3);
gaussMean(4,:) = polyval(c,T);
c = polyfit(T,ORB(:,5),3);
gaussMean(5,:) = polyval(c,T);
c = polyfit(T,ORB(:,6),3);
gaussMean(6,:) = polyval(c,T);

figure 
plot(T/period,kep(1,:))
hold on
plot(T/period,kepMean(1,:),'Color','r','LineWidth',2)
plot(T/period,gaussMean(1,:),'Color','k','LineWidth',2)
title(sprintf('Evolution of the semimajor axis in %d periods', ceil(T(end)/period)))
ylabel("Semi-major axis $[km$]")
xlabel("Periods")
legend("Cartesian propagation","Filtered cartesian propagation","Filtered gauss propagation")

figure 
plot(T/period,kep(2,:))
hold on
plot(T/period,kepMean(2,:),'Color','r','LineWidth',2)
plot(T/period,gaussMean(2,:),'Color','k','LineWidth',2)
title(sprintf('Evolution of the eccentricity in %d periods', ceil(T(end)/period)))
ylabel("Eccentricity")
xlabel("Periods")
legend("Cartesian propagation","Filtered cartesian propagation","Filtered gauss propagation")

figure 
plot(T/period,kep(3,:)*180/pi)
hold on
plot(T/period,kepMean(3,:)*180/pi,'Color','r','LineWidth',2)
plot(T/period,gaussMean(3,:)*180/pi,'Color','k','LineWidth',2)
title(sprintf('Evolution of the inclination in %d periods', ceil(T(end)/period)))
ylabel("Inclination $[deg]$")
xlabel("Periods")
legend("Cartesian propagation","Filtered cartesian propagation","Filtered gauss propagation")

figure 
plot(T/period,kep(4,:)*180/pi)
hold on
plot(T/period,kepMean(4,:)*180/pi,'Color','r','LineWidth',2)
plot(T/period,gaussMean(4,:)*180/pi,'Color','k','LineWidth',2)
title(sprintf('Evolution of the RAAN in %d periods', ceil(T(end)/period)))
ylabel("RAAN $[deg]$")
xlabel("Periods")
legend("Cartesian propagation","Filtered cartesian propagation","Filtered gauss propagation")

figure 
plot(T/period,kep(5,:)*180/pi)
hold on
plot(T/period,kepMean(5,:)*180/pi,'Color','r','LineWidth',2)
plot(T/period,gaussMean(5,:)*180/pi,'Color','k','LineWidth',2)
title(sprintf('Evolution of the argument of pericentre in %d periods', ceil(T(end)/period)))
ylabel("Argument of pericentre $[deg]$")
xlabel("Periods")
legend("Cartesian propagation","Filtered cartesian propagation","Filtered gauss propagation")

%% CARTESIAN VS GAUSS PROPAGATION ERROR

figure();   
plot(T/period,abs(kep(1,:)-ORB(:,1)')./kep(1,:) * 100,'Color','k')
xlabel("Periods")
ylabel("error \%")
title("Semi-major axis relative error")

figure();   
plot(T/period,abs(kep(2,:)-ORB(:,2)')./kep(2,:) * 100,'Color','k')
xlabel("Periods")
ylabel("error \%")
title("Eccentricity  relative error")

figure();   
plot(T/period,abs(kep(3,:)-ORB(:,3)')./kep(3,:) * 100,'Color','k')
xlabel("Periods")
ylabel("error \%")
title("Inclination  relative error")

figure();   
plot(T/period,abs(kep(4,:)-ORB(:,4)')./kep(4,:) * 100,'Color','k')
xlabel("Periods")
ylabel("error \%")
title("RAAN  relative error")

figure();   
plot(T/period,abs(kep(5,:)-ORB(:,5)')./kep(5,:) * 100,'Color','k')
xlabel("Periods")
ylabel("error \%")
title("Argument of pericentre relative error")


%% COMPARISON WITH REAL DATA

%Data chosen from the rocket body of the PSLV-C57 launcher
C57 = importdata("PSVL C-57 Rocket Body.txt");

eReal = C57.data(:,1);
iReal = deg2rad(C57.data(:,3));
raanReal = deg2rad(C57.data(:,4));
omegaReal = deg2rad(C57.data(:,5));
aReal = C57.data(:,10);
fReal = deg2rad(C57.data(:,9));

tspanReal = 0:60*60:60*60*(size(C57.data,1)-1);

orbitRealGuess = [aReal(1),eReal(1),iReal(1),raanReal(1),omegaReal(1),fReal(1)];

[Treal, orbReal] = ode113( @(t,y) eqMotionKep(t, y, @(t,y) PerAccCalculatorKep(t, y, parameters), parameters ), tspanReal, orbitRealGuess, options);

raanReal(raanReal>1) = raanReal(raanReal>1) - 2*pi;
omegaReal(omegaReal<6) = omegaReal(omegaReal<6) + 2*pi;

%% PLOT REAL VS PROPAGATED

figure 
plot(Treal/(3600*24),orbReal(:,1))
grid on
hold on
plot(Treal/(3600*24),aReal)
title(sprintf('Evolution of the semimajor axis of PSLV-C57 in 3 months'))
legend('Theoretical','Real')
xlabel("Days")
ylabel("Semi-major axis $[km$]")

figure 
plot(Treal/(3600*24),orbReal(:,2))
grid on
hold on
plot(Treal/(3600*24),eReal)
title(sprintf('Evolution of the eccentricity of PSLV-C57 in 3 months'))
legend('Theoretical','Real')
xlabel("Days")
ylabel("Eccentricity")

figure 
plot(Treal/(3600*24),rad2deg(orbReal(:,3)))
grid on
hold on
plot(Treal/(3600*24),rad2deg(iReal))
title(sprintf('Evolution of the inclination of PSLV-C57 in 3 months'))
legend('Theoretical','Real')
xlabel("Days")
ylabel("Inclination $[deg]$")

figure 
plot(Treal/(3600*24),rad2deg(orbReal(:,4)))
grid on
hold on
plot(Treal/(3600*24),rad2deg(raanReal))
title(sprintf('Evolution of RAAN of PSLV-C57 in 3 months'))
legend('Theoretical','Real')
xlabel("Days")
ylabel("RAAN $[deg]$")


figure 
plot(Treal/(3600*24),rad2deg(orbReal(:,5))-360)
grid on
hold on
plot(Treal/(3600*24),rad2deg(omegaReal)-360)
title(sprintf('Evolution of the argument of pericentre of PSLV-C57 in 3 months'))
legend('Theoretical','Real')
xlabel("Days")
ylabel("Argument of pericenter $[deg]$")
