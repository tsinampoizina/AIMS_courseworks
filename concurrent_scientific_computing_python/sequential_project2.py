# Project 2: sequential
# Randriamahefasoa Tsinampoizina Marie-Sophie

from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt
import random

N = 5
max_error = 10**(-3)

def int_node(i,j):  
    '''
    return True if the point is an internal node and False if not
    '''
    if i == 0 or i == 6*N or j == 0 or j == 4*N or (0 <= j <= 2*N and i >= 2*N+j) or (3*N <= j <=4*N-1 and i >= 2*N+(4*N-j)): return False
    else: return True

def next_grid(old_U):  
    '''
    modify the values in the grid U to their next values
    '''
    global U
    for i in range(1,6*N): 
        for j in range(1,4*N+1):
            if int_node(i,j): U[j][i] = 0.25*(old_U[j+1][i]+old_U[j-1][i]+old_U[j][i+1]+old_U[j][i-1])

def convergence(old_U,U): 
    '''
    check if the difference between the old grid and the new one is small enough
    '''
    diff = 0.0
    for i in range(1,6*N):
        for j in range(1,4*N):
            diff = diff + abs(U[j][i] - old_U[j][i])
    if diff < max_error: return True
    else: return False

def initialisation():
    '''
    implement the boundary conditions, and set the values of the internal nodes to be 2
    '''
    global U
    
    # boundary conditions
    for j in range(0,4*N+1): U[j][0] = 3*(j*h)          # left
    for j in range(2*N+1,3*N): U[j][6*N] = 12*(j*h-1)   # right

    for i in range(0,2*N+1): U[0][i] = 0                #|
    for j in range(1,2*N): U[j][2*N+j]= 0               #| bottom wall
    for i in range(4*N,6*N+1): U[2*N][i] = 0            #|

    for i in range(3*N,6*N+1): U[3*N][i] = 6            #|
    for j in range(3*N+1,4*N): U[j][2*N+(4*N-j)]= 6     #| top wall
    for i in range(0,2*N+1): U[4*N][i] = 6              #|

    # initial values of the internal nodes
    for i in range(1,6*N): 
        for j in range(1,4*N):
            if int_node(i,j): U[j][i] = 2

def plot():
    '''
    plot the streamlines u(i,j) given the matrix [U(xi,yi)]
    '''
    xi = np.arange(0,3.01,h)
    yi = np.arange(0,2.01,h)
    plt.figure(1)
    plt.contour(xi,yi,U,10,linewidths=0.5,colors='k')
    plt.contourf(xi,yi,U,10,cmap=plt.cm.jet)
    plt.colorbar()
    x = []
    y = []
    for i in range(6*N):
        for j in range(4*N):
            x.append(xi[i])
            y.append(yi[j])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Stream function of the Irrotational flow ")
    plt.scatter(x,y,marker='o',c='b',s=5)
    plt.xlim(0,3)
    plt.ylim(0,2)
    plt.show()

if __name__ == '__main__':
    h = 1./(2*N)
    U = np.zeros((4*N+1,6*N+1),dtype ='float')
    initialisation()
    old_U = U.copy()
    while not convergence(old_U,U):
        old_U = U.copy()
        next_grid(old_U)
    plot()

