function dorb = eqMotionKep(t, orb, PerAccFun, parameters)
% PROTOTYPE
% ds = eqMotion(t, orb, PerAcc, parameters)
%
% If default values are desired insert an empty vector ([]) as the corresponding value
%
% INPUT: 
% t                 [1x1]     Time                                                                                [s]
% Orb               [1x6]     State of the body in keplerian coordinates: Orb = [a e i RAAN omega f]              [km - rad rad rad rad]                                    
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

a = orb(1);
e = orb(2);
i = orb(3);
raan = orb(4);
omega = orb(5);
f = orb(6);

PerAcc = PerAccFun(t,orb);

at = PerAcc(1,1);
an = PerAcc(2,1);
ah = PerAcc(3,1);

%%  USEFUL RELATIONS

E = f2E(f,e);
M = E2M(f,e);

b = a*sqrt(1-e^2);
p = a*(1-e^2);
n = sqrt(mu/a^3);
h = n*a*b;
r = p/(1+e*cos(f));
v = sqrt((2*mu)/r - mu/a);


%% EQUATION OF MOTION

da = (2*a^2*v)/mu*at;

de = 1/v*(2*(e+cos(f))*at-r/a*sin(f)*an);

di = (r*cos(f+omega))/h*ah;

draan = (r*sin(f+omega))/(h*sin(i))*ah;

domega = 1/(e*v)*(2*sin(f)*at+(2*e+r/a*cos(f))*an)-r*sin(f+omega)*cos(i)/(h*sin(i))*ah;

df = h/(r^2)-1/(e*v)*(2*sin(f)*at+(2*e+r/a*cos(f))*an);

dM = n-b/(e*a*v)*(2*(1+e^(2*r)/p)*sin(f)*at+r/a*cos(f)*an);

dorb = [da de di draan domega df]';

end