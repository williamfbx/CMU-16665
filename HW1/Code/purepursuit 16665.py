"""

Path tracking simulation with pure pursuit steering control and PID speed control for 16-665 .

author: Rathin Shah(rsshah), Shruti Gangopadhyay (sgangopa)

"""


import math
import matplotlib.pyplot as plt
import numpy as np

# Pure Pursuit parameters
L = 15  # look ahead distance
dt = 0.1  # discrete time

# Vehicle parameters (m)
LENGTH = 4.5        #length of the vehicle (for the plot)
WIDTH = 2.0         #length of the vehicle (for the plot)
BACKTOWHEEL = 1.0   #length of the vehicle (for the plot)
WHEEL_LEN = 0.3     #length of the vehicle (for the plot)
WHEEL_WIDTH = 0.2   #length of the vehicle (for the plot)
TREAD = 0.7         #length of the vehicle (for the plot)
WB = 2.5            # wheel-base


def plotVehicle(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):

    outline = np.array(
        [
            [
                -BACKTOWHEEL,
                (LENGTH - BACKTOWHEEL),
                (LENGTH - BACKTOWHEEL),
                -BACKTOWHEEL,
                -BACKTOWHEEL,
            ],
            [WIDTH / 2, WIDTH / 2, -WIDTH / 2, -WIDTH / 2, WIDTH / 2],
        ]
    )

    fr_wheel = np.array(
        [
            [WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
            [
                -WHEEL_WIDTH - TREAD,
                -WHEEL_WIDTH - TREAD,
                WHEEL_WIDTH - TREAD,
                WHEEL_WIDTH - TREAD,
                -WHEEL_WIDTH - TREAD,
            ],
        ]
    )

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array(
        [[math.cos(steer), math.sin(steer)], [-math.sin(steer), math.cos(steer)]]
    )

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(
        np.array(outline[0, :]).flatten(), np.array(outline[1, :]).flatten(), truckcolor
    )
    plt.plot(
        np.array(fr_wheel[0, :]).flatten(),
        np.array(fr_wheel[1, :]).flatten(),
        truckcolor,
    )
    plt.plot(
        np.array(rr_wheel[0, :]).flatten(),
        np.array(rr_wheel[1, :]).flatten(),
        truckcolor,
    )
    plt.plot(
        np.array(fl_wheel[0, :]).flatten(),
        np.array(fl_wheel[1, :]).flatten(),
        truckcolor,
    )
    plt.plot(
        np.array(rl_wheel[0, :]).flatten(),
        np.array(rl_wheel[1, :]).flatten(),
        truckcolor,
    )
    plt.plot(x, y, "*")


def getDistance(p1, p2):
    """
    Calculate distance
    :param p1: list, point1
    :param p2: list, point2
    :return: float, distance
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)


class Vehicle:
    def __init__(self, x, y, yaw, vel=0):
        """
        Define a vehicle class (state of the vehicle)
        :param x: float, x position
        :param y: float, y position
        :param yaw: float, vehicle heading
        :param vel: float, velocity

        """
        # State of the vehicle

        self.x = x #x coordinate of the vehicle 
        self.y = y #y coordinate of the vehicle
        self.yaw = yaw  #yaw of the vehicle
        self.vel = vel  #velocity of the vehicle

    def update(self, acc, delta):
        """
        Vehicle motion model, here we are using simple bycicle model
        :param acc: float, acceleration
        :param delta: float, heading control
        """

        # TODO- update the state of the vehicle (x,y,yaw,vel) based on simple bicycle model    
        self.vel += acc * dt
        
        yawdot = (self.vel/WB) * math.tan(delta)
        self.yaw += yawdot * dt
        
        xdot = self.vel * np.cos(self.yaw)
        ydot = self.vel * np.sin(self.yaw)
        self.x += xdot * dt
        self.y += ydot * dt


class Trajectory:
    def __init__(self, traj_x, traj_y):
        """
        Define a trajectory class
        :param traj_x: list, list of x position
        :param traj_y: list, list of y position
        """
        self.traj_x = traj_x
        self.traj_y = traj_y
        self.last_idx = 0

    def getPoint(self, idx):
        return [self.traj_x[idx], self.traj_y[idx]]

    def getTargetPoint(self, pos):
        """
        Get the next look ahead point
        :param pos: list, vehicle position
        :return: list, target point
        """
        target_idx = self.last_idx
        target_point = self.getPoint(target_idx)
        curr_dist = getDistance(pos, target_point)

        while curr_dist < L and target_idx < len(self.traj_x) - 1:
            target_idx += 1
            target_point = self.getPoint(target_idx)
            curr_dist = getDistance(pos, target_point)

        self.last_idx = target_idx
        return self.getPoint(target_idx)


class Controller:
    def __init__(self, kp=1.0, ki=0.1):
        """
        Define a PID controller class
        :param kp: float, kp coeff
        :param ki: float, ki coeff
        :param kd: float, kd coeff
        """
        self.kp = kp
        self.ki = ki
        self.Pterm = 0.0
        self.Iterm = 0.0
        self.last_error = 0.0

    def Longitudinalcontrol(self, error):
        """
        PID main function, given an input, this function will output a acceleration for longitudinal error
        :param error: float, error term
        :return: float, output control
        """
        self.Pterm = self.kp * error
        self.Iterm += error * dt

        self.last_error = error
        output = self.Pterm + self.ki * self.Iterm
        return output
   
    def PurePursuitcontrol(self, error):
        #TODO- find delta        
        delta = np.arctan((2 * np.sin(error) * WB)/L)
        
        return delta

def main():
    # create vehicle
    ego = Vehicle(0, 0, 0)
    plotVehicle(ego.x, ego.y, ego.yaw)

    # target velocity
    target_vel = 10

    # target course
    traj_x = np.arange(0, 100, 0.5)
    traj_y = [math.sin(x / 10.0) * x / 2.0 for x in traj_x]
    traj = Trajectory(traj_x, traj_y)
    goal = traj.getPoint(len(traj_x) - 1)

    # create longitudinal and pure pursuit controller
    PI_acc = Controller()
    PI_yaw = Controller()

    # real trajectory
    traj_ego_x = []
    traj_ego_y = []

    plt.figure(figsize=(12, 8))

    while getDistance([ego.x, ego.y], goal) > 1:
        target_point = traj.getTargetPoint([ego.x, ego.y])

        # use PID to control the speed vehicle
        vel_err = target_vel - ego.vel
        acc = PI_acc.Longitudinalcontrol(vel_err)
        
        # use pure pursuit to control the heading of the vehicle
        # TODO- Calculate the yaw error
        x_diff = target_point[0] - ego.x
        y_diff = target_point[1] - ego.y
        yaw_des = np.arctan2(y_diff, x_diff)
        
        yaw_act = ego.yaw
        
        yaw_err = yaw_des - yaw_act #TODO- Update the equation
        
        delta = PI_yaw.PurePursuitcontrol(yaw_err)  #TODO- update the Pure pursuit controller

        # move the vehicle
        ego.update(acc, delta)

        # store the trajectory
        traj_ego_x.append(ego.x)
        traj_ego_y.append(ego.y)

        # # plots
        plt.cla()
        plt.plot(traj_x, traj_y, "-r", linewidth=5, label="course")
        plt.plot(traj_ego_x, traj_ego_y, "-b", linewidth=2, label="trajectory")
        plt.plot(target_point[0], target_point[1], "og", ms=5, label="target point")
        plotVehicle(ego.x, ego.y, ego.yaw, delta)
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.legend()
        plt.grid(True)
        plt.pause(0.1)


if __name__ == "__main__":
    main()