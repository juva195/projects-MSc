folder = "C:\Users\matte\Documents\Github";
addpath(genpath(folder));

set(groot, 'defaultTextInterpreter', 'latex')
set(groot,'defaultAxesXMinorGrid','on','defaultAxesXMinorGridMode','manual');
set(groot,'defaultAxesYMinorGrid','on','defaultAxesYMinorGridMode','manual');
set(groot, 'defaultLegendLocation', 'northeast')
set(groot, 'defaultLegendInterpreter', 'latex')
set(groot, 'defaultAxesTickLabelInterpreter', 'latex')
set(groot, 'defaultAxesFontWeight', 'bold')
set(groot, 'defaultFigurePosition', [470, 360, 900, 530])
set(groot, 'defaultFigureColormap', turbo(256));
set(groot, 'defaultAxesFontName', 'Palatino Linotype', 'defaultTextFontName', 'Palatino Linotype');
set(groot, 'defaultSurfaceEdgeAlpha', 0.3);
set(groot, 'defaultLineLineWidth', 2);
set(groot, 'defaultFigureColor', [1; 1; 1]);
set(groot, 'defaultAxesColor', 'none');
set(groot, 'defaultAxesFontSize', 20);

default = [];

Options = ["Ocompletion", "yes","setViasual","Elevation"];

%% ODE SETTINGS 
dt = 1;
options = odeset( 'RelTol', 1e-13, 'AbsTol', 1e-14);