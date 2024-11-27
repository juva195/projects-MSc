function [a] = repeatingGroundTrack(m,k,mu,omegaE)
% 
% repeatingGroundTrack is a function that calculates the
% required semi-major axis a in order to have a repeating ground track
% after k satellite revolutions and m Earth revolutions
%
% PROTOTYPE
% [a] = repeatingGroundTrack(m, k, mu, omegaE)
%
% If default values are desired insert an empty vector ([]) as the corresponding value
%
% INPUT:
% m       [1x1]     Number of Earth revolutions (for default m = 1)                                     [-]
% k       [1x1]     Number of orbits to plot (for default k = 10)                                       [-]
% mu      [1x1]     Gravitational parameter of the planet (for default mu = 3.98600433e+05)             [km^3/s^2]
% omegaE  [1x1]     Angular velocity of the planet (for default omegaE = 7.291597763887421e-05)         [rad/s]
%
% OUTPUT:
% a       [1x1]     Semi-major axis required to obtain a repeating ground track                         [km]
%
% -------------------------------------------------------------------------
% DEFAULT SETTINGS

if isempty(m)
    m = 1;
end

if isempty(k)
    k = 10;
end

if isempty(mu)
    mu = astroConstants(13);
end

if isempty(omegaE)
    omegaE = 15.04*pi/(3600*180);
end

%% SEMI-MAJOR AXIS

a = (mu*(m/(omegaE*k))^2)^(1/3);
