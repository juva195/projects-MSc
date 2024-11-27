function Tof = calculateTof(VinfMin, VinfPlus, rp)

muEarth = astroConstants(13);
muSun = astroConstants(4);
REarth = astroConstants(2);

rSOI = REarth * (muEarth/muSun)^(2/5);

eMin = 1+rp*VinfMin^2/muEarth;
ePlus = 1+rp*VinfPlus^2/muEarth;

aMin = -muEarth/VinfMin^2;
aPlus = -muEarth/VinfPlus^2;

thetaMin = acos((rp*(1+eMin)/rSOI-1)/eMin);
thetaPlus = acos((rp*(1+ePlus)/rSOI-1)/ePlus);

EMin = 2*atanh(tan(thetaMin/2)*sqrt((eMin-1)/(eMin+1)));
EPlus = 2*atanh(tan(thetaPlus/2)*sqrt((ePlus-1)/(ePlus+1)));

Tof = (eMin*sinh(EMin)-EMin)*sqrt(-aMin^3/muEarth)+(ePlus*sinh(EPlus)-EPlus)*sqrt(-aPlus^3/muEarth);

end