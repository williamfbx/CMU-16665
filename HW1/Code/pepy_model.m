% Use Runge-Kutta method to solve the ODE
clear;

% Parameters
V = 1;
lf = 1.5;
lr = 1.5;

% Time span
tspan = [0, 10];

% Initial conditions
y0 = [0, 0, 0];

% Run ode45
[t, y] = ode45(@(t, y) odefun(t, y, V, lf, lr), tspan, y0);

% Plot figure
figure;

plot(y(:,1), y(:,2));
title("Cooredinates of vehicle subject to deltaf steering angle")
xlabel("x-axis");
ylabel("y-axis");

% Create function for ODE
% y(t) = [X(t), Y(t), psi(t)]

function dydt = odefun(t, y, V, lf, lr)

    % Front steering angle in radians
    % Part A: Constant
    % deltaf = 1;

    % Part B: Sinusoid normalized with period of 2 seconds
    deltaf = 1 * sin(pi * t);

    % Part C: Square wave normalized with period of 2 seconds
    % deltaf = square(pi * t);

    dydt = zeros(3, 1);
    dydt(1) = V * cos(y(3));
    dydt(2) = V * sin(y(3));
    dydt(3) = (V * tan(deltaf) ) / (lf + lr);
end