function dy = eqMotionCar(t, Car, PerAccFun, parameters)
% PROTOTYPE
% ds = eqMotion(t, orb, PerAcc, parameters)
%
% If default values are desired insert an empty vector ([]) as the corresponding value
%
% INPUT: 
% t                 [1x1]     Time                                                                                [s]
% Car               [1x6]     State of the body in cartesian coordinates: [r v]                                   [km/s km]                                    
% PerAcc            [3x1]     Perturbing Accelerations                                                            [km/s^2]
% Physical Parameters 
%   parameters.mu   [1x1]      Gravitational parameter of the planet (for default mu = 3.98600433e+05)            [km^3/s^2]
%   parameters.J2   [1x1]      Second Armonic of the planet (for default J2 = 0.1082626925638815e-2)              [-]
%   parameters.CD   [1x1]      Drag Coefficient                                                                   [-]
%   parameters.A/M  [1x1]      Area to mass ratio                                                                 [m^2/kg]                                                           
%
% OUTPUT:
% ds                [1x7]      Equation of motion of the state of the body 
%
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

CD = parameters.CD;
AMratio = parameters.AMratio;

r = Car(1:3);
v = Car(4:6);

rnorm = norm(r);

PerAcc = PerAccFun(t,Car);

%% EQUATION OF MOTION

dy = [ v; (-mu/rnorm^3)*r+PerAcc];
