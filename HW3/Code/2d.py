"""
Air Mobility Project- 16665
Author: Guanya Shi
guanyas@andrew.cmu.edu
"""

import numpy as np
import meshcat
import meshcat.geometry as geometry
import meshcat.transformations as tf
import matplotlib.pyplot as plt
from time import sleep
from control import lqr
import argparse

class Quadrotor_2D():
	def __init__(self):
		# parameters
		self.m = 0.027 # kg
		self.J = 8.571710e-5 # inertia
		self.arm = 0.0325 # arm length
		self.g = 9.81 # gravity

		# control actuation matrix
		self.B = np.array([[1, 1],
						   [-self.arm, self.arm]])
		self.B_inv = np.linalg.inv(self.B)

		# noise level
		self.sigma_t = 0.25 # for translational dynamics
		self.sigma_r = 0.25 # for rotational dynamics

		# initial state
		self.p = np.array([0., 0])
		self.v = np.array([0., 0])
		self.a = np.array([0., 0])
		self.theta = 0.
		self.omega = 0.

		# initial control (hovering)
		self.u = np.array([self.m*self.g/2, self.m*self.g/2])

		# control limit for each rotor (N)
		self.umin = 0.
		self.umax = 0.024 * self.g

		# total time and discretizaiton step
		self.dt = 0.01
		self.step = 0
		self.t = 0.

	def reset(self):
		self.sigma_t = 0.25
		self.sigma_r = 0.25
		# self.sigma_t = 0
		# self.sigma_r = 0
		self.p = np.array([0., 0])
		self.v = np.array([0., 0])
		self.a = np.array([0., 0])
		self.theta = 0.
		self.omega = 0.
		self.u = np.array([self.m*self.g/2, self.m*self.g/2])
		self.step = 0
		self.t = 0.

	def dynamics(self, u):
		'''
		Problem A-1: Based on lecture 2, complete the following codes.
		Please only complete the "..." parts. Don't change other codes.
		self.u is the control input (two rotor forces).
		Hint: first convert self.u to total thrust and torque using the control actuation matrix.
		'''
		u = np.clip(u, self.umin, self.umax)
		self.u = np.clip(u, self.umin, self.umax)
  
		U = self.B @ u
		T = U[0]
		tau = U[1]

		pdot = self.v
		vdot = np.array([0, -self.g]) + (T/self.m) * np.array([-np.sin(self.theta), np.cos(self.theta)])
		thetadot = self.omega
		omegadot = tau/self.J

		self.p += self.dt * pdot
		self.v += self.dt * vdot + self.dt * self.sigma_t * np.random.normal(size=2)
		self.a = vdot
		self.theta += self.dt * thetadot
		self.omega += self.dt * omegadot + self.dt * self.sigma_r * np.random.normal()

		self.t += self.dt
		self.step += 1

	def cascaded_control(self, p_d, v_d, a_d, omega_d=0., tau_d=0.):
		'''
		Problem A-2 and A-4: Based on lecture 3, complete the following codes.
		Please only complete the "..." parts. Don't change other codes.
		Your goal is to develop a cascaded controller to track a trajectory (p_d, v_d, a_d).
		omega_d=0, tau_d=0 except for Problem A-5.
		Hint for gain tuning: position control gain is smaller (1-10);
		Attitude control gain is bigger (10-200).
		'''
		# position control
		K_P = 10
		K_D = 5

		# attitude control
		K_Ptau = 200
		K_Dtau = 30
  
		# calculate desired force
		f_d = self.m * (-np.array([0., -self.g]) - K_P * (self.p - p_d) - K_D * (self.v - v_d) + a_d)
		# f_d = self.m * (-np.array([0., -self.g]) - K_P * (self.p - p_d) - K_D * (self.v - 0) + 0)
  
		# calculate desired theta
		theta_d = -np.arctan(f_d[0]/f_d[1])
  
		# calculate T and tau
		T = f_d.T @ np.array([-np.sin(self.theta), np.cos(self.theta)])
		tau = self.J * (-K_Ptau * (self.theta - theta_d) - K_Dtau * (self.omega - omega_d) + tau_d)
  
		# map to T1 and T2
		u = self.B_inv @ np.array([T, tau])
		return u

	def linear_control(self, p_d):
		'''
		Problem A-3: Based on lecture 3, complete the following codes.
		Please only complete the "..." parts. Don't change other codes.
		Your goal is to develop a LQR control based on the linearized model around the hovering condition.
		Hint: use the lqr function in the control library.
		'''

		# calculate gain matrix using lqr
		A = np.array([[0, 0, 1, 0, 0, 0], 
                	  [0, 0, 0, 1, 0, 0],
					  [0, 0, 0, 0, -self.g, 0],
					  [0, 0, 0, 0, 0, 0],
					  [0, 0, 0, 0, 0, 1],
       				  [0, 0, 0, 0, 0, 0]])
  
		B = np.array([[0, 0],
					  [0, 0],
					  [0, 0],
					  [1, 0],
					  [0, 0],
					  [0, 1]])
		Q = np.diag([35, 35, 1, 1, 50, 1])
		R = np.diag([0.5, 0.5])
		K = lqr(A, B, Q, R)[0]
    
		# calculate controls for T and tau
		x = np.array([self.p[0], self.p[1], self.v[0], self.v[1], self.theta, self.omega])
		x_e = np.array([p_d[0], p_d[1], 0, 0, 0, 0])
		U_e = np.array([self.g, 0])
		U = U_e - K @ (x-x_e)

		# update controls for T and tau to account for mass and inertia
		U[0] = U[0] * self.m
		U[1] = U[1] * self.J
		u = self.B_inv @ U
		return u

def plot(time, pos, vel, control, theta, omega, pos_des):
	plt.figure(figsize=(16, 4))
	plt.subplot(1, 4, 1)
	colors = ['tab:blue', 'tab:orange']
	names = ['x', 'y']
	for i in range(2):
		plt.plot(time, pos[:,i], color=colors[i], label=names[i]+" actual")
		plt.plot(time, pos_des[:,i], '--', color=colors[i], label=names[i]+" desired")
	plt.xlabel("time (s)")
	plt.ylabel("pos (m)")
	plt.legend()
	plt.subplot(1, 4, 2)
	plt.plot(time, vel)
	plt.xlabel("time (s)")
	plt.ylabel("vel (m/s)")
	plt.legend(["x", "y"])
	plt.subplot(1, 4, 3)
	plt.plot(time, control)
	plt.xlabel("time (s)")
	plt.ylabel("control (N)")
	plt.legend(["1", "2"])
	plt.subplot(1, 4, 4)
	plt.plot(time, theta)
	plt.plot(time, omega)
	plt.xlabel("time (s)")
	plt.legend(["theta (rad)", "omega (rad/s)"])
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	robot = Quadrotor_2D()
	total_time = 2 * np.pi
	total_step = int(total_time/robot.dt+1)
	time = np.linspace(0, total_time, total_step)
	pos = np.zeros((total_step, 2))
	pos_des = np.zeros((total_step, 2))
	vel = np.zeros((total_step, 2))
	control = np.zeros((total_step, 2))
	control[0,:] = robot.u
	theta = np.zeros(total_step)
	omega = np.zeros(total_step)

	parser = argparse.ArgumentParser()
	parser.add_argument('question', type=int)
	question = parser.parse_args().question
	
	'''
	Problem A-1: system modeling
	'''
	if question == 1:
		robot.sigma_r = 0.
		robot.sigma_t = 0.
		for i in range(21):
			u = np.array([0.019, 0.023]) * 9.81
			robot.dynamics(u)
			if i % 10 == 0:
				print('************************')
				print('pos:', robot.p)
				print('vel:', robot.v)
				print('theta:', robot.theta)
				print('omega:', robot.omega)

	'''
	Problem A-2: cascaded setpoint control
	Complete p_d, v_d, and a_d
	'''
	robot.reset()
	while True:
		if question != 2 or robot.step >= total_step-1:
			break
		p_d = np.array([1, 0])
		v_d = np.array([0, 0])
		a_d = np.array([0, 0])
		u = robot.cascaded_control(p_d, v_d, a_d)
		robot.dynamics(u)
		pos[robot.step,:] = robot.p
		pos_des[robot.step,:] = p_d
		vel[robot.step,:] = robot.v
		control[robot.step,:] = robot.u
		theta[robot.step] = robot.theta
		omega[robot.step] = robot.omega
	if question == 2:
		pos_des[0,:] = p_d
		plot(time, pos, vel, control, theta, omega, pos_des)
  
		# Calculate rise time, maximum overshoot, and average control energy
		x = pos[:,0]
		y = pos[:,1]
  
		final_value_x = p_d[0] 
		threshold_10_x = 0.1 * final_value_x
		threshold_90_x = 0.9 * final_value_x
		time_10_x = time[np.where(x >= threshold_10_x)[0][0]]
		time_90_x = time[np.where(x >= threshold_90_x)[0][0]]
		rise_time_x = time_90_x - time_10_x
     
		final_value_y = p_d[1]
		threshold_10_y = 0.1 * final_value_y
		threshold_90_y = 0.9 * final_value_y
		time_10_y = time[np.where(y >= threshold_10_y)[0][0]]
		time_90_y = time[np.where(y >= threshold_90_y)[0][0]]
		rise_time_y = time_90_y - time_10_y
		print(f"Rise time for x is {round(rise_time_x, 5)}, rise time for y is {round(rise_time_y, 5)}")
  
		max_overshoot_x = abs(max(x)-final_value_x)
		max_overshoot_y = abs(max(y)-final_value_y)
		print(f"Maximum overshoot in x is {round(max_overshoot_x, 5)}, maximum overshoot in y is {round(max_overshoot_y, 5)}")
  
		squared_norms = np.sum(control**2, axis=1)
		avg_energy = np.mean(squared_norms)
		print(f"The average control energy is {round(avg_energy, 5)}")

	'''
	Problem A-3: linear setpoint control
	Complete p_d
	'''
	robot.reset()
	while True:
		if question != 3 or robot.step >= total_step-1:
			break
		p_d = np.array([1, 0])
		u = robot.linear_control(p_d)
		robot.dynamics(u)
		pos[robot.step,:] = robot.p
		pos_des[robot.step,:] = p_d
		vel[robot.step,:] = robot.v
		control[robot.step,:] = robot.u
		theta[robot.step] = robot.theta
		omega[robot.step] = robot.omega
	if question == 3:
		pos_des[0,:] = p_d
		plot(time, pos, vel, control, theta, omega, pos_des)
  
		# Calculate rise time, maximum overshoot, and average control energy
		x = pos[:,0]
		y = pos[:,1]
  
		final_value_x = p_d[0] 
		threshold_10_x = 0.1 * final_value_x
		threshold_90_x = 0.9 * final_value_x
		time_10_x = time[np.where(x >= threshold_10_x)[0][0]]
		time_90_x = time[np.where(x >= threshold_90_x)[0][0]]
		rise_time_x = time_90_x - time_10_x
     
		final_value_y = p_d[1]
		threshold_10_y = 0.1 * final_value_y
		threshold_90_y = 0.9 * final_value_y
		time_10_y = time[np.where(y >= threshold_10_y)[0][0]]
		time_90_y = time[np.where(y >= threshold_90_y)[0][0]]
		rise_time_y = time_90_y - time_10_y
		print(f"Rise time for x is {round(rise_time_x, 5)}, rise time for y is {round(rise_time_y, 5)}")
  
		max_overshoot_x = abs(max(x)-final_value_x)
		max_overshoot_y = abs(max(y)-final_value_y)
		print(f"Maximum overshoot in x is {round(max_overshoot_x, 5)}, maximum overshoot in y is {round(max_overshoot_y, 5)}")
  
		squared_norms = np.sum(control**2, axis=1)
		avg_energy = np.mean(squared_norms)
		print(f"The average control energy is {round(avg_energy, 5)}")

	'''
	Problem A-4: cascaded tracking control
	Complete p_d, v_d, and a_d
	'''
	robot.reset()
	while True:
		if question != 4 or robot.step >= total_step-1:
			break
		t = robot.t
		p_d = np.array([np.sin(t), 0.5 * np.cos(2*t + np.pi/2)])
		v_d = np.array([np.cos(t), -np.sin(2*t + np.pi/2)])
		a_d = np.array([-np.sin(t), -2 * np.cos(2*t + np.pi/2)])
		u = robot.cascaded_control(p_d, v_d, a_d)
		robot.dynamics(u)
		pos[robot.step,:] = robot.p
		pos_des[robot.step,:] = p_d
		vel[robot.step,:] = robot.v
		control[robot.step,:] = robot.u
		theta[robot.step] = robot.theta
		omega[robot.step] = robot.omega
	if question == 4:
		plot(time, pos, vel, control, theta, omega, pos_des)
  
		# calculate RMSE
		rmse = np.sqrt(np.mean((pos_des - pos) ** 2))
		print(f"RMSE: {round(rmse, 5)}")

	'''
	Problem A-5: trajectory generation and differential flatness
	Design trajectory and tracking controllers here.
	'''
	robot.reset()
	while True:
		if question != 5 or robot.step >= total_step-1:
			break
		t = robot.t 

		# calculate trajectory polynomial constants
		T = total_time
		A = np.array([[T**4, T**5, T**6, T**7, T**8],
					  [4*(T**3), 5*(T**4), 6*(T**5), 7*(T**6), 8*(T**7)],
					  [12*(T**2), 20*(T**3), 30*(T**4), 42*(T**5), 56*(T**6)],
       				  [24*T, 60*(T**2), 120*(T**3), 210*(T**4), 336*(T**5)],
             		  [24, 120*T, 360*(T**2), 840*(T**3), 1680*(T**4)]])

		C = np.array([1, 0, 0, 0, 0]).reshape(5, 1)
		B = np.linalg.inv(A) @ C
  
		# calculate reference trajectory x-values (y-values are 0)
		px_r = B[0]*(t**4) + B[1]*(t**5) + B[2]*(t**6) + B[3]*(t**7) + B[4]*(t**8)
		vx_r = B[0]*4*(t**3) + B[1]*5*(t**4) + B[2]*6*(t**5) + B[3]*7*(t**6) + B[4]*8*(t**7)
		ax_r = B[0]*12*(t**2) + B[1]*20*(t**3) + B[2]*30*(t**4) + B[3]*42*(t**5) + B[4]*56*(t**6)
		jx_r = B[0]*24*t + B[1]*60*(t**2) + B[2]*120*(t**3) + B[3]*210*(t**4) + B[4]*336*(t**5)
		sx_r = B[0]*24 + B[1]*120*t + B[2]*360*(t**2) + B[3]*840*(t**3) + B[4]*1680*(t**4)

		p_r = np.array([px_r.item(), 0])
		v_r = np.array([vx_r.item(), 0])
		a_r = np.array([ax_r.item(), 0])
		j_r = np.array([jx_r.item(), 0])
		s_r = np.array([sx_r.item(), 0])

		# naive reference trajectory
		# p_r = np.array([t/T, 0])
		# v_r = np.array([1/T, 0])
		# a_r = np.array([0, 0])
		# j_r = np.array([0, 0])
		# s_r = np.array([0, 0])

		'''
		differential flatness
		'''
  
		# part I
		# --------------------------------------------------------------------------
		# g = np.array([0, -robot.g])
		# y = (robot.a - g)/np.linalg.norm(robot.a - g)
		# theta1 = -np.arctan2(y[0], y[1])
		# Thr = np.dot(robot.a - g, y)
		# x = np.array([np.cos(theta1), np.sin(theta1)])

		# omega_d = -np.dot(j_r, x)/Thr
		# tau_d = -(np.dot(s_r, x) + 2 * np.dot(j_r, y) * robot.omega)/Thr
  
		# u = robot.cascaded_control(p_r, v_r, a_r, omega_d, tau_d)
		# --------------------------------------------------------------------------

		# part II
		# --------------------------------------------------------------------------
		# omega_d = 0
		# tau_d = 0

		# u = robot.cascaded_control(p_r, v_r, a_r, omega_d, tau_d)
		# --------------------------------------------------------------------------
  
		# part III
		# --------------------------------------------------------------------------
		g = np.array([0, -robot.g])
		y_r = (a_r - g)/np.linalg.norm(a_r - g)
		theta_r = -np.arctan2(y_r[0], y_r[1])
		T_r = np.dot(a_r - g, y_r)
		x_r = np.array([np.cos(theta_r), np.sin(theta_r)])
  
		omega_r = -np.dot(j_r, x_r)/T_r
		tau_r = -(np.dot(s_r, x_r) + 2 * np.dot(j_r, y_r) * omega_r)/T_r
		# u = robot.cascaded_control(p_r, v_r, a_r, omega_r, tau_r)
    
		u = robot.B_inv @ np.array([robot.m * T_r, robot.J * tau_r])
		# --------------------------------------------------------------------------

		robot.dynamics(u)
		pos[robot.step,:] = robot.p
		pos_des[robot.step,:] = p_r
		vel[robot.step,:] = robot.v
		control[robot.step,:] = robot.u
		theta[robot.step] = robot.theta
		omega[robot.step] = robot.omega
	if question == 5:
		plot(time, pos, vel, control, theta, omega, pos_des)
  
		# calculate RMSE
		rmse = np.sqrt(np.mean((pos_des - pos) ** 2))
		print(f"RMSE: {round(rmse, 5)}")

	'''
	Animation using meshcat
	'''
	vis = meshcat.Visualizer()
	vis.open()

	vis["/Cameras/default"].set_transform(
		tf.translation_matrix([0,0,0]).dot(
		tf.euler_matrix(0,np.radians(-30),-np.pi/2)))

	vis["/Cameras/default/rotated/<object>"].set_transform(
		tf.translation_matrix([1,0,0]))

	vis["Quadrotor"].set_object(geometry.StlMeshGeometry.from_file('./crazyflie2.stl'))
	
	vertices = np.array([[0,0.5],[0,0],[0,0]]).astype(np.float32)
	vis["lines_segments"].set_object(geometry.Line(geometry.PointsGeometry(vertices), \
									 geometry.MeshBasicMaterial(color=0xff0000,linewidth=100.)))
	
	while True:
		for i in range(total_step):
			vis["Quadrotor"].set_transform(
				tf.translation_matrix([pos[i,0], 0, pos[i,1]]).dot(tf.euler_matrix(0, theta[i], 0)))
			vis["lines_segments"].set_transform(
				tf.translation_matrix([pos[i,0], 0, pos[i,1]]).dot(tf.euler_matrix(0, theta[i], 0)))				
			sleep(robot.dt)