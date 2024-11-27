function [rho] = densityCalculator(r,parameters)
%
% densityCalculator calculates the density of the air at a certain altitude
% provided the parameters of the mission and the radius
%
% PROTOTYPE
% function [rho] = densityCalculator(r,parameters)
%
% If default values are desired insert an empty vector ([]) as the corresponding value
%
% INPUT:
% r          [3xN]      Radius vector of the orbit at each timestamp              [Km]
% parameters            Struct containing all the data for the mission
%
% OUTPUT:
% rho        [1xN]      Density of the air at the altitude for each timestamp
% H          [1xN]      Scale Height for each timestamp
% -------------------------------------------------------------------------

%%  DENSITY DATA

baseAltitude = [0, 25, 30:10:150, 180, 200:50:500, 600:100:1000];

density = [1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3, 3.206e-4, 8.770e-5, 1.905e-5, 3.396e-6, 5.297e-7, 9.661e-8, 2.438e-8, 8.484e-9, 3.845e-9, ...
    2.070e-9, 5.464e-10, 2.789e-10, 7.248e-11, 2.418e-11, 9.158e-12, 3.725e-12, 1.585e-12, 6.967e-13, 1.454e-13, 3.614e-14, 1.170e-14, 5.245e-15, 3.019e-15];

scaleHeight = [7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549, 5.799, 5.382, 5.877, 7.263, 9.473, 12.636, 16.149...
    22.523, 29.740, 37.105, 45.546, 53.628, 53.298, 58.515, 60.828, 63.822, 71.835, 88.667, 124.64, 181.05, 268.00];

%% RHO AND H ESTRACTION

Re = parameters.Re;

rnorm = vecnorm(r,2,1);
height = rnorm-Re;

rho= zeros(1,size(r,2));

for j = 1:size(r,2)
    if height(j) >= 1000
        rho0 = density(end);
        H = scaleHeight(end);
    else
        ind = find(height(j)-baseAltitude>0,1,"last");
        rho0 = density(ind);
        H = scaleHeight(ind);
        h0 = baseAltitude(ind);
    end
    rho(j) = rho0*exp(-(height(j)-h0)/H);
end











