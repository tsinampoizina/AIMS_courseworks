# Randriamahefasoa Tsinampoizina Marie-Sophie
# Sequential Game of Life

import matplotlib.pyplot as plt
from time import sleep
from numpy import zeros
import numpy as np
from pylab import *
import math

steps =100
M = 20     # size of the grid

def next_state(Sji,j,i):
    '''
    modify the state of the cell according to its neighbors following the rule
    '''
    global A
    if A[j,i] == 1 and (Sji<2 or Sji>3): A[j,i] = -1
    if A[j,i]==0 and Sji == 3: A[j,i] = 2

def f(x):
    '''
    this function satisfies f(1)=f(-1)=1 and f(0)=f(2)=0
    '''
    return np.abs(x)*(2-np.abs(x))

def sum_f(a1,a2,a3,a4,a5,a6,a7,a8):
    '''
    sum f(x) for the 8 neighbors x=a,b,...,h
    '''
    ls = [f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8)]
    return int(sum(ls))


def evolve(steps,A):
    for i in range(0,steps):
        for m in range(0,M):
            for n in range(0,M):
                S = sum_f(A[(m-1)%M,n%M], A[(m-1)%M,(n-1)%M], A[(m-1)%M,(n+1)%M], A[(m+1)%M,n%M], A[(m+1)%M,(n-1)%M], A[(m+1)%M,(n+1)%M], A[m%M,(n-1)%M], A[m%M,(n+1)%M])
                if A[m,n] == 1 and (S<2 or S>3): A[m,n] = -1   # don't change it yet to 0, because we need to remember its old value
                if A[m,n]==0 and S == 3: A[m,n] = 2  # don't change it yet to 0, because we need to remember its old value
        for m in range(0,M):
            for n in range(0,M): 
                if A[m,n] == 2: A[m,n] = 1      # now change the 2 to 0
                if A[m,n] == -1: A[m,n] = 0      # now change the -1 to 1
        print 'step',i+1
        plt.imshow(A, interpolation='nearest', axes=ax1, animated=True)
        plt.draw()
        print A,'\n'
        sleep(0.2) 
#    show()


print 'press r for a random grid, and s for a space ship'
res = raw_input()

# initialisation
if res == 's':
    A = np.zeros((M,M),dtype='int')     # matrix definition
    A[1,0],A[2,1],A[0,2],A[1,2],A[2,2]=1,1,1,1,1

if res == 'r':
    A = np.array([np.random.randint(0,2) for i in range(M**2)]).reshape(M,M)

plt.ion()
plt.hold(False)
fig1 = plt.figure(1)
ax1 = fig1.gca()

evolve(steps,A)
