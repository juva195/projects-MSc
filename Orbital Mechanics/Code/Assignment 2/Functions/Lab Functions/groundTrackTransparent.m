function [alpha, delta, lon, lat] = groundTrackTransparent(CarOrOrb, k, dt, thetaG0, omegaE, mu, t0)
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

if isempty(k)
    k = 10;
end

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


%% Initial Parameters 

if height(CarOrOrb) == 1
    a = CarOrOrb(1);
    e = CarOrOrb(2);
    i = CarOrOrb(3);
    Omega = CarOrOrb(4);
    omega = CarOrOrb(5);
    f0 = CarOrOrb(6);
    [r0, v0] = orbitalToCar(a,e,i,Omega,omega,f0,mu);
elseif height(CarOrOrb) == 3
    r0 = CarOrOrb(:,1);
    v0 = CarOrOrb(:,2);
    [a, ~, ~, ~, ~, ~] = carToOrbital(r0,v0,mu);
else
    error('The variable CarOrOrb has the wrong dimensions.')
end

y0 = [ r0; v0];

% Set time span
OrbP = 2*pi*sqrt( a^3/mu);                            % Orbital period [s]
tspan = 0:dt:OrbP*k;
days = OrbP*k/(60*60*24);

% Set options for the ODE solver
options = odeset( 'RelTol', 1e-12, 'AbsTol', 1e-12);

%% Solve the orbit

% Perform the integration
[ T, Y ] = ode113( @(t,y) ode2bp(t,y,mu), tspan, y0, options );

r = Y(:,1:3);
NormR = vecnorm(r');

%% Define Latitude and Longitude

delta = asin(r(:,3)./NormR');
alpha = atan2(r(:,2),r(:,1));

thetaG = omegaE*(T-t0) + thetaG0;

lon = rad2deg(wrapTo2Pi(alpha - thetaG + pi) - pi);         % Longitude
lat = rad2deg(delta);                                       % Latitude

%% PLOT

lat(lon>179) = nan;
plot(lon(2:end-1),lat(2:end-1),"Color","#EDB120",'LineWidth',1.5)
plot(lon(1),lat(1),"LineStyle","none","Marker", "o","Color","k")
plot(lon(end),lat(end),"LineStyle","none","Marker", "*","Color","k")

hl = legend('Perturbed ground track',"","","Unperturbed ground track");
set(hl, 'TextColor','k', 'Color','w', 'EdgeColor','k')
hl.FontSize = 15;

if k == 1
    title(sprintf('Ground Track of one orbit'))
else
    title(sprintf('Ground Track of %d days',round(days)))
end

xlabel('Longitude')
ylabel('Latitude')
xticks(-180:60:180)
yticks(-90:30:90)
xlim([-180 180])
ylim([-90 90])



set(gca,'YDir','normal')





