% DBM with lane change
clear;

% Parameters
Vx = 30;
m = 1573;
Iz = 2873;
lf = 1.1;
lr = 1.58;
Calphaf = 80000;
Calphar = 80000;

% Populate linear state space system of the form xdot = A * x + 
% B1 * deltaf + B2 * psidot_des and y = C * x + D * deltaf
% Populate A matrix
A = zeros(4,4);
A(1,2) = 1;
A(2,2) = -(2 * Calphaf + 2 * Calphar)/(m * Vx);
A(2,3) = (2 * Calphaf + 2 * Calphar)/m;
A(2,4) = (-2 * Calphaf * lf + 2 * Calphar * lr)/(m * Vx);
A(3,4) = 1;
A(4,2) = -(2 * Calphaf * lf - 2 * Calphar * lr)/(Iz * Vx);
A(4,3) = (2 * Calphaf * lf - 2 * Calphar * lr)/Iz;
A(4,4) = -(2 * Calphaf * lf * lf + 2 * Calphar * lr * lr)/(Iz * Vx);

% Populate B1 matrix
B1 = zeros(4,1);
B1(2,1) = (2 * Calphaf)/m;
B1(4,1) = (2 * Calphaf * lf)/Iz;

% Populate B2 matrix
B2 = zeros(4,1);
B2(2,1) = -(2 * Calphaf * lf - 2 * Calphar * lr)/(m * Vx) - Vx;
B2(4,1) = -(2 * Calphaf * lf * lf + 2 * Calphar * lr * lr)/(Iz * Vx);

% Populate C matrix
C = eye(4);

% Populate D matrix
D = zeros(4,1);

% Create state space object for open-loop system
open_loop_sys = ss(A, B1, C, D);

% Check open-loop system eigenvalues
Eopen = eig(A);

% Desired eigenvalues
p = [complex(-10,-5);
    complex(-10,5);
    -30;
    -40];

% Solve for K using pole placement
K = place(A,B1,p);

% Check for closed-loop system eigenvalues
Aclosed = A - B1 * K;
Eclosed = eig(Aclosed);

% Create state space object for closed-loop system
closed_loop_sys = ss(Aclosed, B2, C, D);

% Create inputs for psidot_des
tstep = 0.01;

u1dot = zeros(1, 1/tstep);
u2dot = ones(1, 5/tstep) * (Vx/1000);
u3dot = zeros(1, 1/tstep);
u4dot = ones(1, 5/tstep) * (-Vx/500);

psidot_des = [u1dot u2dot u3dot u4dot];

% Numerically integrate to get psi_des
psi_des = cumtrapz(tstep, psidot_des);

% Create linear simulation object
tIn = 0:tstep:((length(psidot_des) - 1) * tstep);

[y, tOut, x] = lsim(closed_loop_sys, psidot_des, tIn);

% Errors
e1 = x(:,1);
e1dot = x(:,2);
e2 = x(:,3);
e2dot = x(:,4);

% Calculate derivative of steering input deltaf
deltaf = -K * transpose(x);
deltafdot = gradient(deltaf, tstep);

% Plot of derivative of steering input (deltafdot)
figure();
plot(tOut, deltafdot);
title("Derivative of steering input vs. Time")
xlabel("Time (sec)");
ylabel("Derivative of steering input (rad/sec)");

% Plot lateral position error e1
figure();
plot(tOut, e1);
title("Lateral position error vs. Time")
xlabel("Time (sec)");
ylabel("Lateral position error (m)");

% Plot yaw angle error e2
figure();
plot(tOut, e2);
title("Yaw angle error vs. Time")
xlabel("Time (sec)");
ylabel("Yaw angle error (rad)");

% Plot desired and actual vehicle path in global frame
num_tsteps = length(tOut);

% Desired path
X_des = zeros(1,num_tsteps);
Y_des = zeros(1,num_tsteps);
X_des(1) = 0;
Y_des(1) = -5;

% Actual path
X_act = zeros(1,num_tsteps);
Y_act = zeros(1,num_tsteps);
X_act(1) = 0;
Y_act(1) = -5;

% Equations from page 40 of Rajamani (2012)
for step = 2:num_tsteps
    X_des(step) = X_des(step-1) + Vx * tstep * cos(psi_des(step));
    Y_des(step) = Y_des(step-1) + Vx * tstep * sin(psi_des(step));
    X_act(step) = X_des(step) - e1(step) * sin(e2(step) + psi_des(step));
    Y_act(step) = Y_des(step) + e1(step) * cos(e2(step) + psi_des(step));
end

figure();
plot(X_des,Y_des);
hold on
plot(X_act, Y_act);
title("Cooredinates in global frame of desired and actual vehicle path");
legend("Desired Path","Actual Path");
xlabel("x-axis (m)");
ylabel("y-axis (m)");
hold off