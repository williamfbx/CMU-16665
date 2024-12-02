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
from math_utils import *
from scipy.spatial.transform import Rotation
from scipy.linalg import expm
import argparse

class Quadrotor():
	def __init__(self):
		# parameters
		self.m = 0.027 # kg
		self.J = np.diag([8.571710e-5, 8.655602e-5, 15.261652e-5]) # inertia matrix
		self.J_inv = np.linalg.inv(self.J)
		self.arm = 0.0325 # arm length
		self.t2t = 0.006 # thrust to torque ratio
		self.g = 9.81 # gravity

		# control actuation matrix
		self.B = np.array([[1., 1., 1., 1.],
			               [-self.arm, -self.arm, self.arm, self.arm],
			               [-self.arm, self.arm, self.arm, -self.arm],
			               [-self.t2t, self.t2t, -self.t2t, self.t2t]])
		self.B_inv = np.linalg.inv(self.B)
		
		# noise level
		self.sigma_t = 0.25
		self.sigma_r = 0.25
		
		# disturbance and its estimation
		self.d = np.array([0., 0, 0])
		self.d_hat = np.array([0., 0, 0])
  
		# velocity estimation for L1 adaptive control
		self.v_hat = np.array([0., 0, 0])

		# initial state
		self.p = np.array([0., 0, 0])
		self.v = np.array([0., 0, 0])
		self.R = np.eye(3)
		self.q = np.array([1., 0, 0, 0])
		self.omega = np.array([0., 0, 0])
		self.euler_rpy = np.array([0., 0, 0])

		# initial control (hovering)
		self.u = np.array([1, 1, 1, 1]) * self.m * self.g / 4.
		self.T_curr = self.g

		# control limit for each rotor (N)
		self.umin = 0.
		self.umax = 0.012 * self.g

		# total time and discretizaiton step
		self.dt = 0.01
		self.step = 0
		self.t = 0.

	def reset(self):
		self.sigma_t = 0.25
		self.sigma_r = 0.25
		self.d = np.array([0., 0, 0])
		self.p = np.array([0., 0, 0])
		self.v = np.array([0., 0, 0])
		self.R = np.eye(3)
		self.q = np.array([1., 0, 0, 0])
		self.omega = np.array([0., 0, 0])
		self.euler_rpy = np.array([0., 0, 0])
		self.u = np.array([1, 1, 1, 1]) * self.m * self.g / 4.
		self.step = 0
		self.t = 0.
		self.T_curr = self.g

	def dynamics(self, u):
		'''
		Problem B-1: Based on lecture 2, complete the following codes.
		Please only complete the "..." parts. Don't change other codes.
		self.u is the control input (four rotor forces).
		Hint: first convert self.u to total thrust and torque using the control actuation matrix.
		Hint: use the qintegrate function to update self.q
		'''
		u = np.clip(u, self.umin, self.umax)
		self.u = np.clip(u, self.umin, self.umax)
    
		U = self.B @ u
		T = U[0]
		tau1 = U[1]
		tau2 = U[2]
		tau3 = U[3]

		pdot = self.v
		vdot = np.array([0, 0, -self.g]) + T/self.m * (self.R @ np.array([0, 0, 1]))
		omegadot = np.linalg.inv(self.J) @ (np.cross(self.J @ self.omega, self.omega) + np.array([tau1, tau2, tau3]))

		self.p += self.dt * pdot
		self.v += self.dt * vdot + self.dt * (self.sigma_t * np.random.normal(size=3) + self.d) 
		self.q = qintegrate(self.q, self.omega, self.dt)
		self.R = qtoR(self.q)
		self.omega += self.dt * omegadot + self.dt * self.sigma_r * np.random.normal(size=3)
		self.euler_rpy = Rotation.from_matrix(self.R).as_euler('xyz')

		self.t += self.dt
		self.step += 1

	def cascaded_control(self, p_d, v_d, a_d, yaw_d):
		'''
		Problem B-2: Based on lecture 3, complete the following codes.
		Please only complete the "..." parts. Don't change other codes.
		Your goal is to develop a cascaded controller to track a trajectory (p_d, v_d, a_d, yaw_d).
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
		f_d = -np.array([0, 0, -self.g]) - K_P * (self.p - p_d) - K_D * (self.v - v_d) + a_d

		# calculate desired T
		e3 = np.array([0, 0, 1])
		z = self.R @ e3
		T = f_d.T @ z
  
		# calculate desired rotation
		z_d = f_d/np.linalg.norm(f_d)
		n = np.cross(e3, z_d)
		n_unitvec = n/np.linalg.norm(n)
		rho = np.arcsin(np.linalg.norm(n))
  
		R_EB = Rotation.from_rotvec(rho * n_unitvec).as_matrix()
		R_AE = Rotation.from_euler('z', yaw_d).as_matrix()
		R_d = R_AE @ R_EB
  
		# calculate rotation error and alpha
		R_e = R_d.T @ self.R
		alpha = -K_Ptau * vee(R_e - R_e.T) - K_Dtau * self.omega
  
		# calculate tau
		tau = self.J @ alpha - np.cross(self.J @ self.omega, self.omega)
  
		u = self.B_inv @ np.array([self.m * T, tau[0], tau[1], tau[2]])
		return u

	def adaptive_control(self, p_d, v_d, a_d, yaw_d):
		'''
		Problem B-3: Based on lecture 4, implement adaptive control methods.
		For integral control, this function should be same as cascaded_control, 
		with an extra I gain in the position control loop.
		Hint for integral control: you can use self.d_hat to accumlate/integrate the position error.
		'''
  
		# part I integral control
		# --------------------------------------------------------------------------
		# # position control
		# K_P = 10
		# K_D = 5
		# K_I = 10

		# # attitude control
		# K_Ptau = 200
		# K_Dtau = 30
  
		# # using self.d_hat to accumlate position error
		# self.d_hat += (self.p - p_d) * self.dt
  
		# # calculate desired force
		# f_d = -np.array([0, 0, -self.g]) - K_P * (self.p - p_d) - K_D * (self.v - v_d) - K_I * self.d_hat + a_d

		# # calculate desired T
		# e3 = np.array([0, 0, 1])
		# z = self.R @ e3
		# T = f_d.T @ z
  
		# # calculate desired rotation
		# z_d = f_d/np.linalg.norm(f_d)
		# n = np.cross(e3, z_d)
		# n_unitvec = n/np.linalg.norm(n)
		# rho = np.arcsin(np.linalg.norm(n))
  
		# R_EB = Rotation.from_rotvec(rho * n_unitvec).as_matrix()
		# R_AE = Rotation.from_euler('z', yaw_d).as_matrix()
		# R_d = R_AE @ R_EB
  
		# # calculate rotation error and alpha
		# R_e = R_d.T @ self.R
		# alpha = -K_Ptau * vee(R_e - R_e.T) - K_Dtau * self.omega
  
		# # calculate tau
		# tau = self.J @ alpha - np.cross(self.J @ self.omega, self.omega)
  
		# u = self.B_inv @ np.array([self.m * T, tau[0], tau[1], tau[2]])
		# --------------------------------------------------------------------------
  
		# part III L1 control
		# --------------------------------------------------------------------------
		# low pass filter and Hurwitz matrix
		alpha = 0.5
		As = -0.5 * np.eye(3)
  
		# position control
		K_P = 10
		K_D = 5

		# attitude control
		K_Ptau = 200
		K_Dtau = 30
  
		# update velicty predictor
		vdot_hat = np.array([0, 0, -self.g]) + self.T_curr * (self.R @ np.array([0, 0, 1])) + self.d_hat + As @ (self.v_hat - self.v)
		self.v_hat += self.dt * vdot_hat
  
		# update disturbance predictor
		dnew_hat = -np.linalg.inv(expm(self.dt * As) - np.eye(3)) @ As @ expm(self.dt * As) @ (self.v_hat - self.v)
		self.d_hat = alpha * self.d_hat + (1-alpha) * dnew_hat
  
		# calculate desired force
		f_d = -np.array([0, 0, -self.g]) - K_P * (self.p - p_d) - K_D * (self.v - v_d) - self.d_hat + a_d
  
		# calculate desired T
		e3 = np.array([0, 0, 1])
		z = self.R @ e3
		T = f_d.T @ z
  
		# calculate desired rotation
		z_d = f_d/np.linalg.norm(f_d)
		n = np.cross(e3, z_d)
		n_unitvec = n/np.linalg.norm(n)
		rho = np.arcsin(np.linalg.norm(n))
  
		R_EB = Rotation.from_rotvec(rho * n_unitvec).as_matrix()
		R_AE = Rotation.from_euler('z', yaw_d).as_matrix()
		R_d = R_AE @ R_EB
  
		# calculate rotation error and alpha
		R_e = R_d.T @ self.R
		alpha = -K_Ptau * vee(R_e - R_e.T) - K_Dtau * self.omega
  
		# calculate tau and update current T
		tau = self.J @ alpha - np.cross(self.J @ self.omega, self.omega)
		self.T_curr = T
  
		u = self.B_inv @ np.array([self.m * T, tau[0], tau[1], tau[2]])
		# --------------------------------------------------------------------------

		return u

def plot(time, pos, vel, control, euler_rpy, omega, pos_des):
	plt.figure(figsize=(20, 4))
	plt.subplot(1, 5, 1)
	colors = ['tab:blue', 'tab:orange', 'tab:green']
	names = ['x', 'y', 'z']
	for i in range(3):
		plt.plot(time, pos[:,i], color=colors[i], label=names[i]+" actual")
		plt.plot(time, pos_des[:,i], '--', color=colors[i], label=names[i]+" desired")
	plt.xlabel("time (s)")
	plt.ylabel("pos (m)")
	plt.legend()
	plt.subplot(1, 5, 2)
	plt.plot(time, vel)
	plt.xlabel("time (s)")
	plt.ylabel("vel (m/s)")
	plt.legend(["x", "y", "z"])
	plt.subplot(1, 5, 3)
	plt.plot(time, control)
	plt.xlabel("time (s)")
	plt.ylabel("control (N)")
	plt.legend(["1", "2", "3", "4"])
	plt.subplot(1, 5, 4)
	plt.plot(time, euler_rpy)
	plt.xlabel("time (s)")
	plt.legend(["roll (rad)", "pitch (rad)", "yaw (rad)"])
	plt.subplot(1, 5, 5)
	plt.plot(time, omega)
	plt.xlabel("time (s)")
	plt.ylabel("angular rate (rad/s)")
	plt.legend(["x", "y", "z"])
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	robot = Quadrotor()
	total_time = 3 * np.pi
	total_step = int(total_time/robot.dt+1)
	time = np.linspace(0, total_time, total_step)
	pos = np.zeros((total_step, 3))
	pos_des = np.zeros((total_step, 3))
	vel = np.zeros((total_step, 3))
	control = np.zeros((total_step, 4))
	control[0, :] = robot.u
	quat = np.zeros((total_step, 4))
	quat[0, :] = robot.q
	euler_rpy = np.zeros((total_step, 3))
	omega = np.zeros((total_step, 3))

	parser = argparse.ArgumentParser()
	parser.add_argument('question', type=int)
	question = parser.parse_args().question

	'''
	Problem B-1: system modeling
	'''
	if question == 1:
		robot.sigma_r = 0.
		robot.sigma_t = 0.
		for i in range(21):
			u = np.array([0.006, 0.008, 0.010, 0.012]) * 9.81
			robot.dynamics(u)
			if i % 10 == 0:
				print('************************')
				print('pos:', robot.p)
				print('vel:', robot.v)
				print('quaternion:', robot.q)
				print('omega:', robot.omega)

	'''
	Problem B-2: cascaded tracking control
	'''
	robot.reset()
	while True:
		if question != 2 or robot.step >= total_step-1:
			break
		t = robot.t
  
		# part I setpoint tracking
		# p_d = np.array([1, 1, 1])
		# v_d = np.array([0, 0, 0])
		# a_d = np.array([0, 0, 0])
		# yaw_d = 0
		# yaw_d = (np.pi/3) * (t/total_time)
  
		# part II trajectory tracking
		p_d = np.array([np.sin(2*t), np.cos(2*t)-1, 0.5*t])
		v_d = np.array([2*np.cos(2*t), -2*np.sin(2*t), 0.5])
		a_d = np.array([-4*np.sin(2*t), -4*np.cos(2*t), 0])
		yaw_d = 0
  
		u = robot.cascaded_control(p_d, v_d, a_d, yaw_d)
		robot.dynamics(u)
		pos[robot.step,:] = robot.p
		pos_des[robot.step,:] = p_d
		vel[robot.step,:] = robot.v
		control[robot.step,:] = robot.u
		quat[robot.step] = robot.q
		euler_rpy[robot.step] = robot.euler_rpy
		omega[robot.step] = robot.omega
	if question == 2:
		plot(time, pos, vel, control, euler_rpy, omega, pos_des)
  
		# calculate average control energy and RMSE
		squared_norms = np.sum(control**2, axis=1)
		avg_energy = np.mean(squared_norms)
		print(f"The average control energy is {round(avg_energy, 5)}")

		rmse = np.sqrt(np.mean((pos_des - pos) ** 2))
		print(f"RMSE: {round(rmse, 5)}")

	'''
	Problem B-3: integral and adaptive control
	'''
	robot.reset()
	while True:
		if question != 3 or robot.step >= total_step-1:
			break
		t = robot.t
  
		# part I disturbance
		# robot.d = np.array([0.5, 0.5, 1])

		# part II and III disturbance
		robot.d = np.array([0.5, np.sin(t), np.cos(np.sqrt(2)*t)])
  
		# setpoint tracking
		# p_d = np.array([1, 1, 1])
		# v_d = np.array([0, 0, 0])
		# a_d = np.array([0, 0, 0])

		# trajectory tracking
		p_d = np.array([np.sin(2*t), np.cos(2*t)-1, 0.5*t])
		v_d = np.array([2*np.cos(2*t), -2*np.sin(2*t), 0.5])
		a_d = np.array([-4*np.sin(2*t), -4*np.cos(2*t), 0])
  
		yaw_d = 0.
		u = robot.adaptive_control(p_d, v_d, a_d, yaw_d)
		robot.dynamics(u)
		pos[robot.step,:] = robot.p
		pos_des[robot.step,:] = p_d
		vel[robot.step,:] = robot.v
		control[robot.step,:] = robot.u
		quat[robot.step] = robot.q
		euler_rpy[robot.step] = robot.euler_rpy
		omega[robot.step] = robot.omega
	if question == 3:
		plot(time, pos, vel, control, euler_rpy, omega, pos_des)

		# Calculate rise time, maximum overshoot, and RMSE
		x = pos[:,0]
		y = pos[:,1]
		z = pos[:,2]
  
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
  
		final_value_z = p_d[2] 
		threshold_10_z = 0.1 * final_value_z
		threshold_90_z = 0.9 * final_value_z
		time_10_z = time[np.where(z >= threshold_10_z)[0][0]]
		time_90_z = time[np.where(z >= threshold_90_z)[0][0]]
		rise_time_z = time_90_z - time_10_z

		print(f"Rise time for x is {round(rise_time_x, 5)}, rise time for y is {round(rise_time_y, 5)}, rise time for z is {round(rise_time_z, 5)}")
		print(f"Average rise time is {round((rise_time_x + rise_time_y + rise_time_z)/3, 5)}")
  
		max_overshoot_x = abs(max(x)-final_value_x)
		max_overshoot_y = abs(max(y)-final_value_y)
		max_overshoot_z = abs(max(z)-final_value_z)
		print(f"Maximum overshoot in x is {round(max_overshoot_x, 5)}, maximum overshoot in y is {round(max_overshoot_y, 5)}, maximum overshoot in z is {round(max_overshoot_z, 5)}")
  
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
				tf.translation_matrix(pos[i]).dot(tf.quaternion_matrix(quat[i])))
			vis["lines_segments"].set_transform(
				tf.translation_matrix(pos[i]).dot(tf.quaternion_matrix(quat[i])))				
			sleep(robot.dt)