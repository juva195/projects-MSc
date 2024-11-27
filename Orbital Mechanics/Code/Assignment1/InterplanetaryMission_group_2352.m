clear all
close all
clc 
addpath(genpath('functions'));
%%
%{
This program can only be run on windows machines, as it depends on
executable functions.
It should take about 1-2 minutes to run the whole program including the
plots.
For non-windows machines run "NonWindows.m", however that program will take
significantly longer to run.
%}
%% PROBLEM DATA

firstLaunchDate = [2028, 01, 01, 0, 0, 0]; % First possible launch date

lastLaunchDate = [2058, 01, 01, 0, 0, 0]; % Last possible launch date

timeStep = 2; % Time step in days

flybyWindow = 365; % Flyby transfer window in days

interceptWindow = 365; % Asteroid intercept window in days

sunMu = astroConstants(4);
earthMu = astroConstants(13);

%% FIND BEST SOLUTIONS

% Compute
DV = findFlyby_mex(firstLaunchDate, lastLaunchDate, timeStep, flybyWindow, interceptWindow);

%% PLOT SOLUTIONS

daySpan = date2mjd2000(firstLaunchDate):timeStep:date2mjd2000(lastLaunchDate);
dateSpan = [];
for i=1:length(daySpan)
    dateSpan = [dateSpan, datetime(mjd20002date(daySpan(i)))];
end
[minDV, minIdx] = min(DV);
DVlim = minDV+0.3;

figure()
sgtitle("Delta V map")
subplot(5,1,1:3)
plot(dateSpan,DV)
hold on
plot(dateSpan(minIdx),minDV,'o','MarkerSize',10)
yline(DVlim,'LineWidth',1,'Color','r')
xlimSub = xlim;
legend("Transfer delta V", "Global minimum","Arbitrary delta V limit")
ylabel("Velocity $km/s$",'Interpreter','latex')

subplot(5,1,4:5)
DVredacted = DV;
DVredacted(DV>DVlim) = nan;
DVredactedOrdered = sort(DVredacted);
localMins = islocalmin(DVredacted);
plot(dateSpan,DVredacted)
hold on
plot(dateSpan(localMins),DVredacted(localMins),'o','MarkerSize',10)
for i=1:length(localMins)
    if localMins(i) == 1
    textString = sprintf('$%.2fkm/s$ \n %s', DVredacted(i), dateSpan(i));
    text(dateSpan(i)+100, DVredacted(i)-0.1, textString, 'FontSize', 13,'Interpreter','latex','FontWeight','bold');
    end
end
ylim([minDV-0.3, DVlim])
xlim(xlimSub)
ylabel("Velocity $km/s$",'Interpreter','latex')
legend("Transfer delta V", "Local minimums")

%% CHOOSE OPTIMAL SOLUTION

% Choose 23/02/30
chosenIdx = find(localMins==1,1);

% Compute new first and last launch dates
firstLaunchDateOptimal = mjd20002date(daySpan(chosenIdx)-30);
lastLaunchDateOptimal = mjd20002date(daySpan(chosenIdx)+30);
timestepOptimal = 0.25;

[DVoptimal,optDepDV,optIntDV,optGraDV,optFlyDV,mjd2000,departureV,flybyDepartureV,VinfplusVec,VinfminusVec,rpFlyby,delta,VPminus,VPplus] = findFlyby_mex(firstLaunchDateOptimal, lastLaunchDateOptimal, timestepOptimal, flybyWindow, interceptWindow);

%% PLOT OPTIMAL SOLUTION

daySpanOptimal = date2mjd2000(firstLaunchDateOptimal):timestepOptimal:date2mjd2000(lastLaunchDateOptimal);
dateSpanOptimal = [];
for i=1:length(daySpanOptimal)
    dateSpanOptimal = [dateSpanOptimal, datetime(mjd20002date(daySpanOptimal(i)))];
end
[minDVOptimal, chosenIdx] = min(DVoptimal);

fprintf('The chosen transfer has the following ΔV:\n');
fprintf('ΔV total = %f km/s \n', minDVOptimal);
fprintf('ΔV departure = %f km/s \n', optDepDV(chosenIdx));
fprintf('ΔV gravity assist = %f km/s \n', optGraDV(chosenIdx));
fprintf('ΔV arrival = %f km/s \n\n', optIntDV(chosenIdx));

boudaryDates = dateSpanOptimal(islocalmin(abs(DVoptimal - DVlim)));

figure()
plot(dateSpanOptimal,DVoptimal)
yline(DVlim,'LineWidth',1,'Color','r')
hold on
plot(dateSpanOptimal(chosenIdx),minDVOptimal,'o','MarkerSize',10)
textString = sprintf('$%.2fkm/s$ \n %s', minDVOptimal, dateSpanOptimal(chosenIdx));
text(dateSpanOptimal(chosenIdx)+1, minDVOptimal-0.3, textString, 'FontSize', 15,'Interpreter','latex','FontWeight','bold');
xl = xline(boudaryDates(1),'LineWidth',1.5,'Color','k');
xl.LabelVerticalAlignment = 'middle';
xl.LabelHorizontalAlignment = 'center';
xl.Label = string(boudaryDates(1));
xl.FontSize = 15;
xl = xline(boudaryDates(3),'LineWidth',1.5,'Color','k');
xl.LabelVerticalAlignment = 'middle';
xl.LabelHorizontalAlignment = 'center';
xl.Label = string(boudaryDates(3));
xl.FontSize = 15;
ylabel("Velocity $km/s$",'Interpreter','latex')
legend("Trasfer delta V","Arbitrary delta V limit","Minimum")
%%  COMPUTE DEPARTURE, FLYBY AND ARRIVAL PARAMETERS

initialKep = uplanet(mjd2000(chosenIdx,1), 2);
venusT = 2*pi * sqrt(initialKep(1)^3 / sunMu );
[initialR, initialV] = orbitalToCar(initialKep(1), initialKep(2), initialKep(3), initialKep(4), initialKep(5), initialKep(6), sunMu);

flybyKep = uplanet(mjd2000(chosenIdx,2), 3);
earthT = 2*pi * sqrt(flybyKep(1)^3/sunMu);
[flybyR, flybyV] = orbitalToCar(flybyKep(1), flybyKep(2), flybyKep(3), flybyKep(4), flybyKep(5), flybyKep(6), sunMu);

[finalKep, ~, ~, ~] = ephNEO(mjd2000(chosenIdx,3), 29);
asteroidT = 2*pi * sqrt(finalKep(1)^3/sunMu);
[finalR, finalV] = orbitalToCar(finalKep(1), finalKep(2), finalKep(3), finalKep(4), finalKep(5), finalKep(6), sunMu);

%% INTERCEPT 

% Calculate time of flight
Tof = calculateTof(norm(VinfminusVec(:,chosenIdx)), norm(VinfplusVec(:,chosenIdx)), rpFlyby(chosenIdx));

fprintf('The flyby has a periapsis of %f km, with an altitude above Earth of %f km.\n', rpFlyby(chosenIdx), rpFlyby(chosenIdx) - astroConstants(23));
fprintf('The total time spent in the SOI of Earth during the flyby is %f h. \n', Tof/3600);
fprintf('The flyby provides a total delta V of %f km/s, with a relation of gained delta V to consumed delta V of %f\n\n', optFlyDV(chosenIdx), optFlyDV(chosenIdx)/optGraDV(chosenIdx));

% Compute normal to "flyby" plane
vInfMinus = VinfminusVec(:,chosenIdx);
vInfPlus = VinfplusVec(:,chosenIdx);
hDir = cross(vInfMinus,vInfPlus)/norm(cross(vInfMinus,vInfPlus));

% Compute perigee direction and vector
theta = delta(chosenIdx)/2 - pi/2;
rpDir = cos(theta) .* vInfMinus./norm(vInfMinus) + sin(theta) .* cross(hDir,vInfMinus./norm(vInfMinus));
rp = rpDir .* rpFlyby(chosenIdx);

% Compute powered flyby velocities
vpMinus = cross(hDir,rpDir) .* VPminus(chosenIdx);
vpPlus = cross(hDir,rpDir) .* VPplus(chosenIdx);

% ODE settings
dt = 10;
options = odeset( 'RelTol', 1e-13, 'AbsTol', 1e-14, 'Events',@soiReached);
y0InterceptMinus = [rp', -vpMinus'];
y0InterceptPlus = [rp', vpPlus'];
tspan = 0:dt:10000;

% Perform the integration - minus
[~, yInterceptMinus] = ode113( @(t,y) ode2bp(t,y,earthMu), tspan, y0InterceptMinus, options);
rInterceptMinus = yInterceptMinus(:,1:3);
vInterceptMinus = yInterceptMinus(:,4:6);

% Perform the integration - plus
[~, yInterceptPlus] = ode113( @(t,y) ode2bp(t,y,earthMu), tspan, y0InterceptPlus, options);
rInterceptPlus = yInterceptPlus(:,1:3);
vInterceptPlus = yInterceptPlus(:,4:6);

% Stich together
rIntercept = [flip(rInterceptMinus); rInterceptPlus];
vIntercept = [flip(vInterceptMinus); vInterceptPlus];

%% VENUS, EARTH AND OBJECT 29

% Settings
dt = 3600;
options = odeset( 'RelTol', 1e-13, 'AbsTol', 1e-14);

% Venus
tspanVenus = 0:dt:venusT;
y0Venus = [initialR', initialV'];
[~, yVenus] = ode113( @(t,y) ode2bp(t,y,sunMu), tspanVenus, y0Venus, options);
rVenus = yVenus(:,1:3);
vVenus = yVenus(:,4:6);

% Earth
tspanEarth = 0:dt:earthT;
y0Earth = [flybyR', flybyV'];
[~, yEarth] = ode113( @(t,y) ode2bp(t,y,sunMu), tspanEarth, y0Earth, options);
rEarth = yEarth(:,1:3);
vEarth = yEarth(:,4:6);

% Object 29
tspanAsteroid = 0:dt:asteroidT;
y0Asteroid = [finalR', finalV'];
[~, yAsteroid] = ode113( @(t,y) ode2bp(t,y,sunMu), tspanAsteroid, y0Asteroid, options);
rAsteroid = yAsteroid(:,1:3);
vAsteroid = yAsteroid(:,4:6);

%% FIRST LEG

tFirstLeg = (mjd2000(chosenIdx,2) - mjd2000(chosenIdx,1)) * 86400;
tspanFirstLeg = 0:dt:tFirstLeg;
y0FirstLeg = [initialR', departureV(:,chosenIdx)'];
[~, yFirstLeg] = ode113( @(t,y) ode2bp(t,y,sunMu), tspanFirstLeg, y0FirstLeg, options);
rFirstLeg = yFirstLeg(:,1:3);
vFirstLeg = yFirstLeg(:,4:6);

%% SECOND LEG

tSecondLeg = (mjd2000(chosenIdx,3) - mjd2000(chosenIdx,2)) * 86400;
tspanSecondLeg = 0:dt:tSecondLeg;
y0SecondLeg = [flybyR', flybyDepartureV(:,chosenIdx)'];
[~, ySecondLeg] = ode113( @(t,y) ode2bp(t,y,sunMu), tspanSecondLeg, y0SecondLeg, options);
rSecondLeg = ySecondLeg(:,1:3);
vSecondLeg = ySecondLeg(:,4:6);

%% Transfer orbital parameters

[a1,e1,i1,Omega1,omega1,theta1_dep] = carToOrbital(rFirstLeg(1,:),vFirstLeg(1,:),sunMu);
[~,~,~,~,~,theta1_arr] = carToOrbital(rFirstLeg(end,:),vFirstLeg(end,:),sunMu);

[a2,e2,i2,Omega2,omega2,theta2_dep] = carToOrbital(rSecondLeg(1,:),vSecondLeg(1,:),sunMu);
[~,~,~,~,~,theta2_arr] = carToOrbital(rSecondLeg(end,:),vSecondLeg(end,:),sunMu);

fprintf('The orbital parameters of the transfer arcs are:\n');
fprintf('In the order [a, e, i, Omega, omega, theta_dep, theta_arr]\n');
fprintf('1st leg: %f km, %f, %f °, %f °, %f °, %f °, %f °\n', a1, e1, i1*180/pi, Omega1*180/pi, omega1*180/pi, theta1_dep*180/pi, theta1_arr*180/pi);
fprintf('2nd leg: %f km, %f, %f °, %f °, %f °, %f °, %f °\n', a2, e2, i2*180/pi, Omega2*180/pi, omega2*180/pi, theta2_dep*180/pi, theta2_arr*180/pi);


%% FIRST LEG PLOT

figure();
background("Stars");
hold on;
opts.Units = "km*5";
planet3D("Sun",opts);
colormap("autumn")
plot3(rVenus(:,1),rVenus(:,2),rVenus(:,3),'--','LineWidth',1)
plot3(rEarth(:,1),rEarth(:,2),rEarth(:,3),'--','LineWidth',1)
patch([rFirstLeg(:,1); nan], [rFirstLeg(:,2); nan], [rFirstLeg(:,3); nan], [vecnorm(vFirstLeg,2,2); nan] ,'LineWidth',2,'FaceColor','none','EdgeColor','interp');
plot3(initialR(1),initialR(2),initialR(3),'o','MarkerSize',10,'MarkerEdgeColor',[0 0.4470 0.7410])
plot3(flybyR(1),flybyR(2),flybyR(3),'o','MarkerSize',10,'MarkerEdgeColor',[0.8500 0.3250 0.0980])
cb = colorbar('Color','w','FontSize',22,'position',[.82, .2, .015, .5],'TickLabelInterpreter','latex');
cb.Label.String = "Velocity $km/s$";
cb.Label.Interpreter = "latex";
clim([min(vecnorm(vFirstLeg,2,2)), max(vecnorm(vFirstLeg,2,2))])
legend("","Venus orbit","Earth orbit","First leg","Venus at launch","Earth at flyby",'Color',"k",'TextColor',"w")
view(-42, 19);
axis normal equal

%% SECOND LEG PLOT

figure();
background("Stars");
hold on;
planet3D("Sun",opts);
colormap("autumn")
plot3(rEarth(:,1),rEarth(:,2),rEarth(:,3),'--','LineWidth',1)
plot3(rAsteroid(:,1),rAsteroid(:,2),rAsteroid(:,3),'--','LineWidth',1)
patch([rSecondLeg(:,1); nan], [rSecondLeg(:,2); nan], [rSecondLeg(:,3); nan], [vecnorm(vSecondLeg,2,2); nan] ,'LineWidth',2,'FaceColor','none','EdgeColor','interp');
plot3(flybyR(1),flybyR(2),flybyR(3),'o','MarkerSize',10,'MarkerEdgeColor',[0 0.4470 0.7410])
plot3(finalR(1),finalR(2),finalR(3),'o','MarkerSize',10,'MarkerEdgeColor',[0.8500 0.3250 0.0980])
cb = colorbar('Color','w','FontSize',22,'position',[.82, .2, .015, .5],'TickLabelInterpreter','latex');
cb.Label.String = "Velocity $km/s$";
cb.Label.Interpreter = "latex";
clim([min(vecnorm(vSecondLeg,2,2)), max(vecnorm(vSecondLeg,2,2))])
legend("","Earth orbit","Target asteroid orbit","Second leg","Earth at flyby","Asteroid at arrival",'Color',"k",'TextColor',"w")
view(153, 45)
axis normal

%% COMPLETE PLOT

figure();
background("Stars")
hold on;
planet3D("Sun",opts);
plot3(rVenus(:,1),rVenus(:,2),rVenus(:,3),'--','LineWidth',1)
plot3(rEarth(:,1),rEarth(:,2),rEarth(:,3),'--','LineWidth',1)
plot3(rAsteroid(:,1),rAsteroid(:,2),rAsteroid(:,3),'--','LineWidth',1)
plot3(rFirstLeg(:,1),rFirstLeg(:,2),rFirstLeg(:,3),'LineWidth',2)
plot3(rSecondLeg(:,1),rSecondLeg(:,2),rSecondLeg(:,3),'LineWidth',2)
plot3(initialR(1),initialR(2),initialR(3),'o','MarkerSize',10,'MarkerEdgeColor',[0 0.4470 0.7410])
plot3(flybyR(1),flybyR(2),flybyR(3),'o','MarkerSize',10,'MarkerEdgeColor',[0.8500 0.3250 0.0980])
plot3(finalR(1),finalR(2),finalR(3),'o','MarkerSize',10,'MarkerEdgeColor',[0.9290 0.6940 0.1250])
planet3D("Sun");
legend("","Venus orbit","Earth orbit","Target asteroid orbit","First leg","Second leg","Venus at departure","Earth at flyby","Asteroid at arrival",'Color',"k",'TextColor',"w")
view(-55, 12)
axis normal equal

%% INTERCEPT PLOT

% Compute directions
minusDir = vInfMinus./norm(vInfMinus);
plusDir = vInfMinus./norm(vInfMinus);

% Plot intercept
figure();
background("Stars");
hold on;
planet3D('Earth Cloudy');
colormap("autumn")
patch([rIntercept(:,1); nan], [rIntercept(:,2); nan], [rIntercept(:,3); nan], [vecnorm(vIntercept,2,2); nan] ,'LineWidth',2,'FaceColor','none','EdgeColor','interp');
cb = colorbar('Color','w','FontSize',22,'position',[.8, .25, .015, .5],'TickLabelInterpreter','latex');
cb.Label.String = "Velocity $km/s$";
cb.Label.Interpreter = "latex";
clim([min(vecnorm(vIntercept,2,2)) max(vecnorm(vIntercept,2,2))])
view(100, 5)
axis tight equal

%% UTILS 

function [value, isterminal, direction] = soiReached(~, y)

% Define the condition to stop integration
thresholdValue =  50000;
stopCondition = (thresholdValue - norm(y(1:3))) > 0;  % Replace with your desired condition

% Set the event criteria
value = stopCondition;      % When value becomes zero, the event is triggered
isterminal = 1;             % Halt integration
direction = 0;              % Detect events in either direction

end

