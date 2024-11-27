function dy = ode2bp(~, y, mu)
%% INPUTS

r = y(1:3);
v = y(4:6);

%% SOLUTION

rNorm = norm(r);
dy=[v; (-mu/rNorm^3)*r];
end