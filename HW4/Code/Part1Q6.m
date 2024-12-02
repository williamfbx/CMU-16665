% Legged Mobility
% Part 1 Q6
% Author: Boxiang Fu
clear;

g = 9.81;
y0 = 1.0;

v0 = linspace(0, 5, 100);
p = linspace(-0.5, 0.5, 100);

% Create meshgrid
[v0_mesh, p_mesh] = meshgrid(v0, p);

% Capture point calculation
xT = -p_mesh + sqrt(p_mesh.^2 + (y0 * v0_mesh.^2) / g);

% Plot
figure;
surf(v0_mesh, p_mesh, xT, 'EdgeColor', 'none');

xlabel('Initial Velocity (v_0) [m/s]', 'FontSize', 12);
ylabel('Center of Pressure (p) [m]', 'FontSize', 12);
zlabel('Capture Point (x_T) [m]', 'FontSize', 12);
title('Capture Point as a Function of Initial Velocity and Center of Pressure', 'FontSize', 14);

colorbar;
view(-135, 30);
grid on;