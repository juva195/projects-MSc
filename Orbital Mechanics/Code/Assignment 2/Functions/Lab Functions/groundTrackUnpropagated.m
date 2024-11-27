function [alpha, delta, lon, lat] = groundTrackUnpropagated(y0, finalTime, parameters, dt, thetaG0, omegaE, mu, t0)
%
% groundTrack plots the ground track of specific orbits given its initial
% parameters and astronomical costants.
%
% PROTOTYPE
% [alpha, delta, lon, lat] = groundTrack(CarOrOrb, k, dt, thetaG0, omegaE, mu, t0)
%
% If default values are desired insert an empty vector ([]) as the corresponding value
%
% INPUT:
% CarOrOrb can be:
% - Car   [3x2]     State of the body in cartesian cordinates: Car = [r; v]                             [km; km/s]
% - Orb   [1x6]      State of the body in keplerian coordinates: Orb = [a e i Omega omega theta]        [km - rad rad rad rad]
% k       [1x1]     Number of orbits to plot (for default k = 10)                                       [-]
% dt      [1x1]     Discretization step (for default dt = 1)                                           [s]
% thetaG0 [1x1]     Initial Greenwhich meridinan (for default thetaG0 = 0)                              [rad]
% omegaE  [1x1]     Angular velocity of the planet (for default omegaE = 7.291597763887421e-05)         [rad/s]
% mu      [1x1]     Gravitational parameter of the planet (for default mu = 3.98600433e+05)             [km^3/s^2]
% t0      [1x1]     Initial Greenwhich meridian time (for default t0 = 0)                               [s]
%
% OUTPUT:
% alpha   [(OrbP*k/dt)x1]    Vector of alpha angle               [rad]
% delta   [(OrbP*k/dt)x1]    Vector of delta angle               [rad]
% lon     [(OrbP*k/dt)x1]    Vector of longitude angle           [rad]
% lat     [(OrbP*k/dt)x1]    Vector of latitude angle            [rad]
%
% -------------------------------------------------------------------------
% DEFAULT SETTINGS

if isempty(dt)
    dt = 1;
end

if isempty(thetaG0)
    thetaG0 = 0;
end

if isempty(omegaE)
    omegaE = 15.04*pi/(3600*180);
end

if isempty(mu)
    mu = astroConstants(13);
end

if isempty(t0)
    t0 = 0;
end

%% PROPAGATE

tspan = 0:dt:finalTime;

% Set options for the ODE solver
options = odeset( 'RelTol', 1e-12, 'AbsTol', 1e-13);

% Perform the integration
[T, y] = ode113( @(t,y) eqMotionCar(t, y, @(t,y) PerAccCalculatorCar(t, y, parameters), parameters ), tspan, y0, options);
r = y(:,1:3);

%% Define Latitude and Longitude

rNorm = vecnorm(r');

delta = asin(r(:,3)./rNorm');
alpha = atan2(r(:,2),r(:,1));

thetaG = omegaE*(T-t0) + thetaG0;

lon = rad2deg(wrapTo2Pi(alpha - thetaG + pi) - pi);         % Longitude
lat = rad2deg(delta);                                       % Latitude

%% PLOT

figure

% Earth = imread("Earth Map GreyWhite.png");
Earth = imread("EarthTexture.jpg");
Earth = flip(Earth);
image([-180;180],[-90;90],Earth)
hold on
grid off
 
lat(lon>179) = nan;
plot(lon(2:end-1),lat(2:end-1),"Color","#D95319",'LineWidth',2)
plot(lon(1),lat(1),"LineStyle","none","Marker", "o","Color","#A2142F")
plot(lon(end),lat(end),"LineStyle","none","Marker", "*","Color","#A2142F")

hl = legend('Ground Track','Start','End');
set(hl, 'TextColor','k', 'Color','w', 'EdgeColor','k')
hl.FontSize = 15;

if finalTime <= 2.1830e+04
    title(sprintf('Ground Track of one orbit'))
else
    title(sprintf('Ground Track of %d days',round(T(end)/(3600*24))))
end

xlabel('Longitude')
ylabel('Latitude')
xticks(-180:60:180)
yticks(-90:30:90)
xlim([-180 180])
ylim([-90 90])



set(gca,'YDir','normal')





