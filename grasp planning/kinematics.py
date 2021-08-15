#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from math import *
import cmath


import math
"""Kinematics Docstring
   This python script provides solution of forward kinematics and 
   inverse kinematics for Universal Robot UR3/5/10.
"""

# Congifuration
# Select the robot type.
# UR10 for 'UR10'
# UR5 for 'UR5'
# UR3 for 'UR3'

ROBOT = 'UR10'


# DH Parameter

if ROBOT == 'UR10':

    # d (unit: mm)
    d1 = 0.1273
    d2 = d3 = 0
    d4 = 0.163941
    d5 = 0.1157
    d6 = 0.0922

    # a (unit: mm)
    a1 = a4 = a5 = a6 = 0
    a2 = -0.612
    a3 = -0.5723

elif ROBOT == 'UR5':

    # d (unit: mm)
    d1 = 0.089159 
    d2 = d3 = 0
    d4 = 0.10915
    d5 = 0.09465
    d6 = 0.0823

    # a (unit: mm)
    a1 = a4 = a5 = a6 = 0
    a2 = -0.425
    a3 = -0.39225

elif ROBOT == 'UR3':

    # d (unit: mm)
    d1 = 0.1519 
    d2 = d3 = 0
    d4 = 0.11235
    d5 = 0.08535
    d6 = 0.0819

    # a (unit: mm)
    a1 = a4 = a5 = a6 = 0
    a2 = -0.24365
    a3 = -0.21325


# List type of D-H parameter
# Do not remove these
d = np.array([d1, d2, d3, d4, d5, d6]) # unit: mm
a = np.array([a1, a2, a3, a4, a5, a6]) # unit: mm
alpha = np.array([pi/2, 0, 0, pi/2, -pi/2, 0]) # unit: radian

def fwd_kin(theta, i_unit='r', o_unit='n'):
    """Solve the HTM based on a list of joint values.

    Args:
        theta: A list of joint values. (unit: radian)
        i_unit: Output format. 'r' for radian; 'd' for degree.
        o_unit: Output format. 'n' for np.array; 'p' for ROS Pose.

    Returns:
        The HTM of end-effector joint w.r.t. base joint
    """

    T_06 = np.matrix(np.identity(4))

    if i_unit == 'd':
        theta = [radians(i) for i in theta]
    
    for i in range(6):
        T_06 *= HTM(i, theta)
    print(T_06)


def select(q_sols, q_d, w=[1]*6):
    """Select the optimal solutions among a set of feasible joint value 
       solutions.

    Args:
        q_sols: A set of feasible joint value solutions (unit: radian)
        q_d: A list of desired joint value solution (unit: radian)
        w: A list of weight corresponding to robot joints

    Returns:
        A list of optimal joint value solution.
    """

    error = []
    for q in q_sols:
        error.append(sum([w[i] * (q[i] - q_d[i]) ** 2 for i in range(6)]))
    
    return q_sols[error.index(min(error))]
    
def range_joint(val,minval,maxval):
        calvalue=val
        if val<minval:
    	    v1=(minval-val)/(2*math.pi)
    	    v2=(maxval-val)/(2*math.pi)
    	    iv2=math.floor(v2)
    	    iv1=math.ceil(v1)
    	    if iv1==iv2:
    	        calvalue=val+2*math.pi*iv1
    	    else:
    	        calvalue=float("inf")
        elif val>maxval:
    	    v1=(val-maxval)/(2*math.pi)
    	    v2=(val-minval)/(2*math.pi)
    	    iv2=math.floor(v2)
    	    iv1=math.ceil(v1)
    	    if iv1==iv2:
    	        calvalue=val-2*math.pi*iv1
    	    else:
    	        calvalue=float("inf")
        else:
            a=1   		
        return calvalue
def select_better(q_sols, q_d, w=[1]*6):
    """Select the optimal solutions among a set of feasible joint value 
       solutions.

    Args:
        q_sols: A set of feasible joint value solutions (unit: radian)
        q_d: A list of desired joint value solution (unit: radian)
        w: A list of weight corresponding to robot joints

    Returns:
        A list of optimal joint value solution.
    """
    q_sols_limit=np.zeros((8, 6))
    j=0
    for ql in q_sols:
        q_sols_limit[j][0]=ql[0]
        q_sols_limit[j][1]=range_joint(ql[1],-5*math.pi/6,-math.pi/6)
        q_sols_limit[j][2]=range_joint(ql[2],-5*math.pi/6,5*math.pi/6)
        q_sols_limit[j][3]=range_joint(ql[3],-7*math.pi/6,math.pi/3)
        q_sols_limit[j][4]=ql[4]
        q_sols_limit[j][5]=ql[5]
        j=j+1
    q_sols_limit.tolist()
    error = []
    for q in q_sols_limit:
        error.append(sum([w[i] * (q[i] - q_d[i]) ** 2 for i in range(3)]))
    
    return q_sols_limit[error.index(min(error))]
    
def HTM(i, theta):
    """Calculate the HTM between two links.

    Args:
        i: A target index of joint value. 
        theta: A list of joint value solution. (unit: radian)

    Returns:
        An HTM of Link l w.r.t. Link l-1, where l = i + 1.
    """

    Rot_z = np.matrix(np.identity(4))
    Rot_z[0, 0] = Rot_z[1, 1] = cos(theta[i])
    Rot_z[0, 1] = -sin(theta[i])
    Rot_z[1, 0] = sin(theta[i])

    Trans_z = np.matrix(np.identity(4))
    Trans_z[2, 3] = d[i]

    Trans_x = np.matrix(np.identity(4))
    Trans_x[0, 3] = a[i]

    Rot_x = np.matrix(np.identity(4))
    Rot_x[1, 1] = Rot_x[2, 2] = cos(alpha[i])
    Rot_x[1, 2] = -sin(alpha[i])
    Rot_x[2, 1] = sin(alpha[i])

    A_i = Rot_z * Trans_z * Trans_x * Rot_x
	    
    return A_i




# Inverse Kinematics

def inv_kin(T_06, q_d, i_unit='r', o_unit='r'):
    """Solve the joint values based on an HTM.

    Args:
        p: A pose.
        q_d: A list of desired joint value solution 
             (unit: radian).
        i_unit: Output format. 'r' for radian; 'd' for degree.
        o_unit: Output format. 'r' for radian; 'd' for degree.

    Returns:
        A list of optimal joint value solution.
    """


    if i_unit == 'd':
        q_d = [radians(i) for i in q_d]

    # Initialization of a set of feasible solutions
    theta = np.zeros((8, 6))

 
    # theta1
    P_05 = T_06[0:3, 3] - d6 * T_06[0:3, 2]
    phi1 = atan2(P_05[1], P_05[0])
    phi2 = acos(d4 / sqrt(P_05[0] ** 2 + P_05[1] ** 2))
    theta1 = [pi / 2 + phi1 + phi2, pi / 2 + phi1 - phi2]
    theta[0:4, 0] = theta1[0]
    theta[4:8, 0] = theta1[1]
  
    # theta5
    P_06 = T_06[0:3, 3]
    theta5 = []
    for i in range(2):
        theta5.append(acos((P_06[0] * sin(theta1[i]) - P_06[1] * cos(theta1[i]) - d4) / d6))
    for i in range(2):
        theta[2*i, 4] = theta5[0]
        theta[2*i+1, 4] = -theta5[0]
        theta[2*i+4, 4] = theta5[1]
        theta[2*i+5, 4] = -theta5[1]
  
    # theta6
    T_60 = np.linalg.inv(T_06)
    theta6 = []
    for i in range(2):
        for j in range(2):
            s1 = sin(theta1[i])
            c1 = cos(theta1[i])
            s5 = sin(theta5[j])
            theta6.append(atan2((-T_60[1, 0] * s1 + T_60[1, 1] * c1) / s5, (T_60[0, 0] * s1 - T_60[0, 1] * c1) / s5))
    for i in range(2):
        theta[i, 5] = theta6[0]
        theta[i+2, 5] = theta6[1]
        theta[i+4, 5] = theta6[2]
        theta[i+6, 5] = theta6[3]

    # theta3, theta2, theta4
    for i in range(8):  
        # theta3
        T_46 = HTM(4, theta[i]) * HTM(5, theta[i])
        T_14 = np.linalg.inv(HTM(0, theta[i])) * T_06 * np.linalg.inv(T_46)
        P_13 = T_14 * np.array([[0, -d4, 0, 1]]).T - np.array([[0, 0, 0, 1]]).T
        if i in [0, 2, 4, 6]:
            theta[i, 2] = -cmath.acos((np.linalg.norm(P_13) ** 2 - a2 ** 2 - a3 ** 2) / (2 * a2 * a3)).real
            theta[i+1, 2] = -theta[i, 2]
        # theta2
        theta[i, 1] = -atan2(P_13[1], -P_13[0]) + asin(a3 * sin(theta[i, 2]) / np.linalg.norm(P_13))
        # theta4
        T_13 = HTM(1, theta[i]) * HTM(2, theta[i])
        T_34 = np.linalg.inv(T_13) * T_14
        theta[i, 3] = atan2(T_34[1, 0], T_34[0, 0])       

    theta = theta.tolist()

    # Select the most close solution
    q_sol = select_better(theta, q_d)

    # Output format
    if o_unit == 'r': # (unit: radian)
        return q_sol
    elif o_unit == 'd': # (unit: degree)
        return [degrees(i) for i in q_sol]

