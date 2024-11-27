function PerAcc = PerAccCalculatorCar(~, Car, parameters)
% PROTOTYPE
% PerAcc = PerAccCalculatorKep(t, orb, parameters);
%
% If default values are desired insert an empty vector ([]) as the corresponding value
%
% INPUT: 
% t                 [1x1]      Time                                                                               [s]
% Car               [1x6]      State of the body in Cartesian coordinates: [r v]                                  [km/s km]                                                                                              
% Physical Parameters 
%   parameters.mu   [1x1]      Gravitational parameter of the planet (for default mu = 3.98600433e+05)            [km^3/s^2]
%   parameters.J2   [1x1]      Second Armonic of the planet (for default J2 = 0.1082626925638815e-2)              [-]
%   parameters.CD   [1x1]      Drag Coefficient                                                                   [-]
%   parameters.A/M  [1x1]      Area to mass ratio                                                                 [m^2/kg]                                                           
%   parameters.Re   [1x1]      Earth's Radius    (for default Re = 3.98600433e+05)                                [km]                                                           
%
%
% OUTPUT:
% PerAcc            [3x1]      Perturbating acceleration                                                          [km/s^2]
%

if isempty(parameters.mu)
    mu = astroConstants(13);
else
    mu = parameters.mu;
end

if isempty(parameters.J2)
    J2 = astroConstants(9);
else
    J2 = parameters.J2;
end

if isempty(parameters.Re)
    Re = astroConstants(23);
else
    Re = parameters.Re;
end

if isempty(parameters.omegaE)
    omegaE = 7.291597763887421e-05;
else
    omegaE = parameters.omegaE;
end

CD = parameters.CD;
AMratio = parameters.AMratio;


r = Car(1:3);
v = Car(4:6);

x = r(1);
y = r(2);
z = r(3);

rnorm = norm(r);

wE = [0; 0; omegaE];


%% SECOND HARMONIC 

vec = [x/rnorm*(5*(z^2)/(rnorm^2)-1); y/rnorm*(5*(z^2)/(rnorm^2)-1); z/rnorm*(5*(z^2)/(rnorm^2)-3)];

aJ2 = 3/2*J2*mu*(Re^2)/(rnorm^4)*vec;

%% AIR DRAG

% DENSITY DATA

baseAltitude = [0, 25, 30:10:150, 180, 200:50:500, 600:100:1000];

density = [1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3, 3.206e-4, 8.770e-5, 1.905e-5, 3.396e-6, 5.297e-7, 9.661e-8, 2.438e-8, 8.484e-9, 3.845e-9, ...
    2.070e-9, 5.464e-10, 2.789e-10, 7.248e-11, 2.418e-11, 9.158e-12, 3.725e-12, 1.585e-12, 6.967e-13, 1.454e-13, 3.614e-14, 1.170e-14, 5.245e-15, 3.019e-15];

scaleHeight = [7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549, 5.799, 5.382, 5.877, 7.263, 9.473, 12.636, 16.149...
    22.523, 29.740, 37.105, 45.546, 53.628, 53.298, 58.515, 60.828, 63.822, 71.835, 88.667, 124.64, 181.05, 268.00];

% RHO AND H ESTRACTION

height = rnorm-Re;

if height >= 1000
        rho0 = density(end);
        H = scaleHeight(end);
        h0 = baseAltitude(end);
else
        ind = find(height-baseAltitude>0,1,"last");
        rho0 = density(ind);
        H = scaleHeight(ind);
        h0 = baseAltitude(ind);
end

rho = rho0*exp(-(height-h0)/H);

vrel = v - cross(wE,r);

aDrag = -1/2*AMratio*CD*rho*(norm(vrel)*1000)^2*vrel/norm(vrel);

%% TOTAL ACCELERATION
PerAcc = aJ2+aDrag/1000;