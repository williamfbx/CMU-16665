% Legged Mobility
% Part 2 Q4
% Author: Boxiang Fu
clear;

% Parameters
M = 80; % mass of robot (kg)
g = 9.81; % gravitational acceleration (m/s^2)
k = 20000; % spring stiffness (N/m)
r = 0.05; % rack and pinion radius (m)
J_m = 0.000506; % motor inertia (kg m^2)
N = 40; % gear ratio

% PID control parameters
kp_outer = 2000; % proportional gain for outer loop
kd_outer = 50; % derivative gain for outer loop
ki_outer = 5; % integral gain for outer loop

kp_inner = 20; % proportional gain for inner loop
kd_inner = 1; % derivative gain for inner loop

% Initial conditions
y0 = 1; % nominal height (m)
y = y0; % initial height (m)
ydot = 0; % initial velocity (m/s)
theta_m = 0; % initial motor angle (rad)
thetadot_m = 0; % initial motor angular velocity (rad/s)
int_error_y = 0; % integral of height error

tau_m_max = 1.36; % motor torque limit (N m)
lambda = 0.05; % low pass filter on integral term

% Desired conditions
y_des = 0.9; % desired height (m)

% Simulation parameters
dt = 0.0001; % time step (s)
outer_loop_steps = 5; % refresh rate between inner and outer loop
t_final = 500; % simulation duration (s)
time = 0:dt:t_final;

% Initialize variables
y_values = zeros(size(time));
tau_m_values = zeros(size(time));
theta_m_values = zeros(size(time));
F_s_des_values = zeros(size(time));

% Thermal motor dynamics
% Parameters
R_th1 = 1.82; % winding-housing thermal resistance (K/W)
R_th2 = 1.78; % housing-environment thermal resistance (K/W)
alpha_cu = 0.0039; % thermal resistance of copper
R_25 = 0.844; % electrical resistance at room temperature (ohm)
k_m = 0.231; % torque constant (Nm/A)
T_amb = 25; % ambient temperature (C);
tau_th = 54.3; % winding thermal time constant (s);

% Variables
T = 25; % initial temperature
T_values = zeros(size(time));

for i = 1:length(time)

    % Outer loop
    if mod(i-1, outer_loop_steps) == 0
        e_y = y_des - y;
        int_error_y = (1 - lambda) * int_error_y + e_y * (outer_loop_steps * dt);
        F_s_des = kp_outer * e_y - kd_outer * ydot + ki_outer * int_error_y + M * g;
    end

    % Inner loop
    delta_l_des = F_s_des / k;
    delta_l_m_des = delta_l_des - (y0 - y);
    theta_m_des = (N / r) * delta_l_m_des;

    e_theta = theta_m_des - theta_m;
    tau_m = kp_inner * e_theta - kd_inner * thetadot_m;
    % Clamp motor torque so it stays within operating limits of Maxon EC90
    tau_m = min(max(tau_m, -tau_m_max), tau_m_max);

    % Motor dynamics
    F_s = k * ((y0 - y) + (r / N) * theta_m);
    thetaddot_m = (tau_m - (r/N) * F_s)/ J_m;
    thetadot_m = thetadot_m + thetaddot_m * dt;
    theta_m = theta_m + thetadot_m * dt;

    % Robot dynamics
    yddot = (F_s - M * g) / M;
    ydot = ydot + yddot * dt;
    y = y + ydot * dt;

    % Save results
    y_values(i) = y;
    tau_m_values(i) = tau_m;
    theta_m_values(i) = theta_m;
    F_s_des_values(i) = F_s_des;

    % Thermal dynamics
    I_mot = tau_m / k_m;
    R = R_25 * (1 + alpha_cu * (T_amb - 25));

    deltaT_max = ((R_th1 + R_th2) * R * I_mot^2) / (1 - alpha_cu * (R_th1 + R_th2) * R * I_mot^2);
    deltaT = deltaT_max * (1 - exp(-time(i)/tau_th));
    T = T_amb + deltaT;
    T_values(i) = T;
end

% Plot results
figure;
subplot(3, 1, 1);
plot(time, y_values);
xlabel('Time (s)');
ylabel('Height (m)');
title('Robot Height');

subplot(3, 1, 2);
plot(time, tau_m_values);
xlabel('Time (s)');
ylabel('Motor Torque (Nm)');
title('Motor Torque');

subplot(3, 1, 3);
plot(time, T_values);
xlabel('Time (s)');
ylabel('Motor Temperature (C)');
title('Motor Temperature');

% subplot(4, 1, 3);
% plot(time, theta_m_values);
% xlabel('Time (s)');
% ylabel('Motor Angle (rad)');
% title('Motor Angle');
% 
% subplot(4, 1, 4);
% plot(time, F_s_des_values);
% xlabel('Time (s)');
% ylabel('Desired Spring Force (N)');
% title('Desired Spring Force');
