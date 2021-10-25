# Randriamahefasoa Tsinampoizina Marie-Sophie
# Root-Mean-Square using the double recursive algorithm

from __future__ import division
import numpy as np
import mpi4py.MPI as MPI
import random
import math
import sys

def read(file_name): 
    '''
     reads the data in the file and returns the array of vector
    '''
    f = open(file_name,"r")
    vec = f.readlines() 
    f.close()
    return np.array(vec,float)

def compute_sigma(x): 
    '''
     sigma returns the sum of squares of the elements in a vector
    '''
    return sum([xi**2 for xi in x])
    
def mu_inverse(p,i): 
    ''' 
    return the global index (in the big vector) corresponding to the local index i
    '''
    return p*sub+i


if __name__ == '__main__':
    status = MPI.Status()
    p = MPI.COMM_WORLD.Get_rank()
    NP = MPI.COMM_WORLD.Get_size()

    xtilde  = read(sys.argv[1])
    M = len(xtilde)
    
    if NP == 1: print 'Root-Mean-Square:', math.sqrt(compute_sigma(xtilde)/M)  
    
    else:
        # subdivisions 
        sub = int(math.ceil(M/NP))
        Mp = range(0,M,sub)
        Mp.append(M)
        I = [Mp[u+1]-Mp[u] for u in range(0,len(Mp)-1)]   # length of the subdivision 
        x = np.zeros(I[p],dtype='float')    # create the subvector of each process
        
        D = int(math.log(NP,2))
        for i in range(I[p]): x[i] = xtilde[mu_inverse(p,i)]  # initialise the subvector
        
        
        for d in range(0,D):
            sigma = compute_sigma(x)
            mes = np.array([sigma],dtype='float')
            MPI.COMM_WORLD.Send([sigma,MPI.FLOAT],dest = (p+2**d)%NP,tag = p)
            mes = np.zeros(1,dtype='float')
            MPI.COMM_WORLD.Recv([mes,MPI.FLOAT], (p+2**d)%NP, (p+2**d)%NP ,status)
            s = mes[0]
            sigma = sigma+s
        print 'Root-Mean-Square', math.sqrt(sigma/M)
