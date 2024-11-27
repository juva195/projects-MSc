function PerAcc = PerAccCalculatorKep(~, orb, parameters)
% PROTOTYPE
% PerAcc = PerAccCalculatorKep(t, orb, parameters);
%
% If default values are desired insert an empty vector ([]) as the corresponding value
%
% INPUT: 
% t                 [1x1]      Time                                                                               [s]
% Orb               [1x6]      State of the body in keplerian coordinates: Orb = [a e i RAAN omega f]             [km - rad rad rad rad]                                                                                              
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

a = orb(1);
e = orb(2);
i = orb(3);
raan = orb(4);
omega = orb(5);
f = orb(6);

[R,V] = orbitalToCar(a,e,i,raan,omega,f,mu);

rnorm = norm(R);
wE = [0;0;omegaE];

p = a*(1-e^2);
r = p/(1+e*cos(f));
v = sqrt((2*mu)/r - mu/a);
n = sqrt(mu/a^3);
b = a*sqrt(1-e^2);
h = n*a*b;

%% PERTURBED ACCELERATION 

vec = [1-3*sin(i)^2*sin(f+omega)^2; sin(i)^2*sin(2*(f+omega)); sin(2*i)*sin(f+omega)];

ematrix =  [e*sin(f) -(1+e*cos(f)); 1+e*cos(f) e*sin(f)];

% J2 perturbed acceleration in RSW frame
aJ2rsw = -3/2*J2*mu*Re^2/(r^4)*vec;

ars = aJ2rsw(1:2);

atn = inv(ematrix).*p.*v./h*ars;

aJ2tnh = [atn(1); atn(2); aJ2rsw(3)];

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

vrel = V - cross(wE,R);

aDragECI = -1/2*AMratio*CD*rho*(norm(vrel)*1000)^2*vrel/norm(vrel);

tver = V/norm(V);

hvec = cross(R,V);
hver = hvec/norm(hvec);

nver = cross(hver,tver);

RotMatrix = [tver, nver, hver];

aDragTnh = transpose(RotMatrix)*aDragECI;

%% TOTAL ACCELERATION
PerAcc = aJ2tnh+aDragTnh/1000;