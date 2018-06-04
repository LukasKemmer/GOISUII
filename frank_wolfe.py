#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 17:58:08 2018

@author: lukaskemmer
"""

import numpy as np
import matplotlib.pyplot as plt

# set seed
np.random.seed(1001001)

##########################################################
# Load data
##########################################################

data = np.load("signal_data.npz")
y = data['y']
t = data['t']
n = len(y)
m = 1
y = y.reshape((n,1))

##########################################################
# Set functions
##########################################################

# Define matrix Ls
L = np.eye(n)[:-1,:] - np.column_stack((np.zeros((n-1,1)), np.eye(n-1)))

# Set A:=1/2LL^T, b=-Ly (use -1 because LD is max problem)
A = 1/2*np.dot(L,L.T)
b = -np.dot(L,y).reshape(n-1, 1)

# Define h and gradient of h
#h2 = lambda x : -1/4*np.dot(np.dot(np.dot(x.T, L), L.T), x) + np.dot(np.dot(x.T, L), y)
h = lambda x : 1/2 * np.dot(np.dot(x.T, A), x) + np.dot(b.T, x)
grad_h = lambda x : np.dot(A,x)+b

##########################################################
# Define optimizer
##########################################################

def frank_wolfe(h, grad_h, epsilon, alpha, A, b):
    # Check that valid alpha is chosen
    assert alpha>0

    # Solve initial problem Q
    mu_k = (np.random.rand(n-1)*alpha).reshape(n-1,1)
    k = 0
    y_k = -alpha * np.sign(np.dot(A, mu_k) + b)
    d_k = y_k - mu_k
    
    # Stop criterion
    while np.dot(grad_h(mu_k).T, d_k / np.sqrt(np.sum(np.power(d_k,2)))) <= -epsilon:
        # Find optimal stepsize
        c = np.dot(grad_h(mu_k).T, d_k)
        Q = np.dot(np.dot(d_k.T, A), d_k)
        t_k = 1
        if Q > 0 and -c <= Q:
            t_k = -c/Q
            
        # Update mu_k
        mu_k = mu_k + t_k*d_k
        
        # Update k
        k = k+1
        
        # Solve Q
        y_k = -alpha * np.sign(np.dot(A, mu_k) + b)
        
        # Update d_k
        d_k = y_k - mu_k
    
    return mu_k, k

##########################################################
# Set parameters
##########################################################

# Set epsilon
epsilon = 1e-2

# Set alpha
alphas = [1/10, 1/2, 1, 5]

# Get results
for i, alpha in enumerate(alphas):
    print("Alpha: {}".format(alpha))
    # Get mu_bar from LD
    mu_bar, n_iter = frank_wolfe(h, grad_h, epsilon, alpha, A, b)
    
    # Get x_k of RL_1 from mu_bar 
    x_bar = y - 1/2*np.dot(L.T, mu_bar)
    
    # Plot results
    plt.plot(t, y)
    plt.plot(t, x_bar)
    plt.title("Alpha={}".format(alpha))
    plt.legend(['Original data (y)', 'Denoised signal (x_bar)'])
    plt.xlabel("t")
    plt.ylabel("")
    plt.show()

'''assert alpha>0

# Solve initial problem Q
mu_k = (np.random.rand(n-1)*alpha).reshape(n-1,1)
k = 0
y_k = -alpha * np.sign(np.dot(A, mu_k) + b)
d_k = y_k - mu_k

# Stop criterion
while np.dot(grad_h(mu_k).T, d_k / np.sqrt(np.sum(np.power(d_k,2)))) <= -epsilon:
    # Find optimal stepsize
    c = np.dot(grad_h(mu_k).T, d_k)
    Q = np.dot(np.dot(d_k.T, A), d_k)
    t_k = 1
    if Q > 0 and -c <= Q:
        t_k = -c/Q
        
    # Update mu_k
    mu_k = mu_k + t_k*d_k
    
    # Update k
    k = k+1
    
    # Solve Q
    y_k = -alpha * np.sign(np.dot(A, mu_k) + b)
    
    # Update d_k
    d_k = y_k - mu_k

mu_bar = mu_k
'''

