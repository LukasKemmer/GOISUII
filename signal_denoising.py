#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 16:28:36 2018

@author: lukaskemmer
"""
import numpy as np
import matplotlib.pyplot as plt

##########################################################
# Load data
##########################################################

data = np.load("signal_data.npz")
y = data['y']
t = data['t']
n = len(y)
m = 1

##########################################################
# 2.2 a) Solve optimization problem 
##########################################################

# Set alpha parameters (> 0)
alphas = [1, 5, 10, 50]

# Define matrices R:=L, b:=y, A:=I
R = np.eye(n)[:-1,:] - np.column_stack((np.zeros((n-1,1)), np.eye(n-1)))
b = y.reshape((n,m))
A = np.eye(n)

# Set plot style
plt.style.use(['seaborn-talk'])

for i, alpha in enumerate(alphas):
    print("Alpha: {}".format(alpha))
    # Calculate x_bar by solving (A^TA + alpha R^TR)x = A^Tb
    x_bar = np.linalg.solve(np.dot(A.T, A) + alpha*np.dot(R.T, R),
                            np.dot(A.T, b))
    # Plot result
    plt.plot(t, y)
    plt.plot(t, x_bar)
    plt.title("Alpha={}".format(alpha))
    plt.legend(['Original data (y)', 'Denoised signal (f(t))'])
    plt.xlabel("t")
    plt.ylabel("")
    plt.show()
    # We can see, that alpha controls the "smotheness" of the results