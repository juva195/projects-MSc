function [DV,optDepDV,optIntDV,optGraDV,optFlyDV,mjd2000,departureV,flybyDepartureV, Vinfplus, Vinfminus, rpFlyby, delta, VPminus, VPplus] = findFlyby(firstLaunchDate, lastLaunchDate, timeStep, flybyWindow, interceptWindow)
%{
This function is compiled in findFlyby_mex
%}

arguments
    firstLaunchDate (1,6) double
    lastLaunchDate (1,6) double
    timeStep (1,1) double
    flybyWindow (1,1) double
    interceptWindow (1,1) double
end

rEarth = astroConstants(23);

muEarth = astroConstants(13);

sunMu = astroConstants(4);

% Set launch dates and intercept windows

possibleLaunchMjd2000 = date2mjd2000(firstLaunchDate):timeStep:date2mjd2000(lastLaunchDate); % All posible launch dates in mjd2000

% Iterate each possible launch date

launchDates = size(possibleLaunchMjd2000,2);

DV = ones(1,launchDates)*100;
optDepDV = zeros(1,launchDates);
optIntDV = zeros(1,launchDates);
optGraDV = zeros(1,launchDates);
optFlyDV = zeros(1,launchDates);
departureV = zeros(3, launchDates);
flybyDepartureV = zeros(3, launchDates);
Vinfplus = zeros(3, launchDates);
Vinfminus = zeros(3, launchDates);
rpFlyby = zeros(1, launchDates);
delta = zeros(1, launchDates);
VPminus = zeros(1, launchDates);
VPplus = zeros(1, launchDates);

mjd2000 = zeros(launchDates,3);

% Iterate launch dates

parfor launchIndex = 1:launchDates
    
    launchMjd2000 = possibleLaunchMjd2000(launchIndex);

    % Get launch orbit

    [launchKep, ~] = uplanet(launchMjd2000, 2); % Keplerian elements at launch
    
    [launchR, launchV] = orbitalToCar(launchKep(1), launchKep(2), launchKep(3), launchKep(4), launchKep(5), launchKep(6), sunMu);
    
    % Iterate flyby transfer arrival dates
    
    for flybyMjd2000 = (launchMjd2000+1):timeStep:(flybyWindow+launchMjd2000) % Minimum TOF is TPAR

        [flybyKep, ~] = uplanet(flybyMjd2000, 3); % Keplerian elements of Earth at flyby
        
        [flybyR, flybyV] = orbitalToCar(flybyKep(1), flybyKep(2), flybyKep(3), flybyKep(4), flybyKep(5), flybyKep(6),  sunMu);
        
        flybyTOF = (flybyMjd2000 - launchMjd2000)*86400;
        
        [~,~,~,ERROR,flybyTransferDepartureV,flybyTransferArrivalV,~,~] = lambertMR(launchR,flybyR,flybyTOF,sunMu,0,0,0,0);
        if ERROR ~= 0
            continue;
        end
        flybyTransferDepartureV = flybyTransferDepartureV';
        flybyTransferArrivalV = flybyTransferArrivalV';

        departureDV = norm(launchV-flybyTransferDepartureV);

        if departureDV>12
            continue;
        end
        
        % Iterate possible intercept dates
        
        for interceptMjd2000 = (flybyMjd2000+1):timeStep:(flybyMjd2000+interceptWindow)

            [interceptKep, ~, ~, ~] = ephNEO(interceptMjd2000,29); % Keplerian elements of Asteroid at intercept
            
            [interceptR, interceptV] = orbitalToCar(interceptKep(1), interceptKep(2), interceptKep(3), interceptKep(4), interceptKep(5), interceptKep(6),  sunMu);
            
            interceptTOF = (interceptMjd2000 - flybyMjd2000) * 86400;
            
            [~,~,~,ERROR,interceptDepartureV,interceptArrivalV,~,~] = lambertMR(flybyR,interceptR,interceptTOF,sunMu,0,0,0,0);
            if ERROR ~= 0
                continue;
            end
            interceptDepartureV = interceptDepartureV';
            interceptArrivalV = interceptArrivalV';
            
            interceptDV = norm(interceptV - interceptArrivalV);
            
            if (interceptDV+departureDV)>12
                continue;
            end

            VinfMinus = flybyTransferArrivalV - flybyV;
            VinfPlus = interceptDepartureV - flybyV;
            
            CosDelta = max(min(dot(VinfMinus,VinfPlus)/(norm(VinfMinus)*norm(VinfPlus)),1),-1);
            Delta = real(acosd(CosDelta));
            
            fun = @(x) Delta*pi/180 - asin(1/(1+(x*(norm(VinfMinus))^2)/muEarth)) - asin(1/(1+(x*(norm(VinfPlus))^2)/muEarth));
            
            if fun(rEarth*145) * fun(rEarth+300) > 0
                continue;
            end
            
            rp = fzero(fun,[rEarth+300, rEarth*145]);
            
            aMinus = - muEarth/(norm(VinfMinus))^2;
            aPlus = - muEarth/(norm(VinfPlus))^2;
            
            VpMinus = sqrt(muEarth*(2/rp-1/aMinus));
            VpPlus = sqrt(muEarth*(2/rp-1/aPlus));
            
            gravityAsistDV = abs(VpPlus-VpMinus);
            
            flybyDV = norm(interceptDepartureV-flybyTransferArrivalV);
            
            auxDV = departureDV + interceptDV + gravityAsistDV;
            
            if auxDV < DV(launchIndex)
                VPminus(launchIndex) = VpMinus;
                VPplus(launchIndex) = VpPlus;
                delta(launchIndex) = Delta;
                Vinfplus(:, launchIndex) = VinfPlus;
                Vinfminus(:, launchIndex) = VinfMinus;
                rpFlyby(launchIndex) = rp;
                DV(launchIndex) = auxDV;
                optDepDV(launchIndex) = departureDV;
                optIntDV(launchIndex) = interceptDV;
                optGraDV(launchIndex) = gravityAsistDV;
                optFlyDV(launchIndex) = flybyDV;
                mjd2000(launchIndex,:) = [launchMjd2000, flybyMjd2000, interceptMjd2000];
                departureV(:, launchIndex) = flybyTransferDepartureV;
                flybyDepartureV(:, launchIndex) = interceptDepartureV;
            end
        end
    end
end

DV(DV==0) = NaN;
DV(DV==100) = NaN;

end