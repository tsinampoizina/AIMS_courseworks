# Project 2: concurrent irrotational fluid flow (with the help of Fortunat Rajaona)
# Randriamahefasoa Tsinampoizina Marie-Sophie
from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import mpi4py.MPI as MPI

N = 10
max_error = 10**(-2)
comm = MPI.COMM_WORLD

def int_node(x,y):  
    '''
    returns True if the point is an internal node and False if not
    '''
    if x == 0 or x == 6*N or y == 0 or y == 4*N or (0 <= y <= 2*N and x >= 2*N+y) or (3*N <= y <=4*N-1 and x >= 2*N+(4*N-y)): return False
    else: return True

def col_int_nodes(x): 
    '''
     counts the internal nodes in the column x
    '''
    count = 0
    for y in range(1,4*N):
        if int_node(x,y): count = count + 1
    return count

def all_int_nodes():
    '''
     count all internal nodes
    '''
    all_int_nodes = 0 
    for x in range(1,6*N):
        all_int_nodes = all_int_nodes + col_int_nodes(x)
    return all_int_nodes

def boundary_exchange(local_U,I):
    global U_right,U_left
    # send right
    if  rank < NP-1:
        mes = local_U[:,I[rank]-1].copy()
        comm.Send([mes,MPI.FLOAT],dest = rank+1, tag = rank)
    # recv left
    if 1 <= rank :
        mes = np.zeros(4*N+1,dtype ='float')
        comm.Recv([mes,MPI.FLOAT],source = rank-1,tag = rank-1)
        U_left = mes

    # send left
    if 1 <= rank :
        mes = local_U[:,0].copy()
        comm.Send([mes,MPI.FLOAT],dest = rank-1,tag = rank)
    # recv right
    if rank < NP-1:
        mes = np.zeros(4*N+1,dtype ='float')
        comm.Recv([mes,MPI.FLOAT],source = rank+1,tag = rank+1)
        U_right = mes

def global_indices(x,y):
    return x+M[rank],y

def update():
    '''
    compute the next grid
    '''
    global local_U
    old_local_U = local_U.copy()
    for x in range(1,I[rank]-1): 
        for y in range(1,4*N):
            u,v = global_indices(x,y)
            if int_node(u,v): local_U[y][x] = 0.25*(old_local_U[y+1][x]+old_local_U[y-1][x]+old_local_U[y][x+1]+old_local_U[y][x-1])
    for x in ([I[rank]-1]):
        for y in range(1,4*N+1):
            u,v = global_indices(x,y)
            if rank == NP - 1:
                if int_node(u,v): local_U[y][x] = 0.25*(old_local_U[y+1][x]+old_local_U[y-1][x]+U[y][6*N]+U[v][u-1])
            elif rank != NP - 1:
                if int_node(u,v): local_U[y][x] = 0.25*(old_local_U[y+1][x]+old_local_U[y-1][x]+U_right[y]+U[v][u-1])
    for x in ([0]):
        for y in range(1,4*N+1):
            u,v = global_indices(x,y)
            if rank == 0: 
                if int_node(u,v): local_U[y][x] = 0.25*(old_local_U[y+1][x]+old_local_U[y-1][x]+U[v][u+1]+U[y][0])
            elif rank != 0 :
                if int_node(u,v): local_U[y][x] = 0.25*(old_local_U[y+1][x]+old_local_U[y-1][x]+U[v][u+1]+U_left[y])
    

def comp_loc_err(old_local_U,local_U,I): 
    '''
    return the local error
    '''
    diff = 0.0
    for x in range(0,I[rank]):
        for y in range(1,4*N):
            u,v = global_indices(x,y)
            if int_node(u,v): diff = diff + abs(local_U[y][x] - old_local_U[y][x])
    return diff

def run(): 
    '''
    compute the next grid U from its old values until the error || U - Uold || < max_error
    '''
    global local_U
    
    total_error = 1
    while total_error > max_error:
        if rank < NP: 
            old_local_U = local_U.copy()
            boundary_exchange(local_U,I)
            update()
            local_error = comp_loc_err(old_local_U,local_U,I)
        
        # send the local_error to process 0
        if 1 <= rank < NP: 
            mes = np.array([local_error],dtype ='float')
            comm.Send([mes,MPI.FLOAT],dest = 0,tag = 10)
        if rank == 0:
            total_error = local_error
            for source in range(1,NP):
                mes = np.zeros(1,dtype='float')
                comm.Recv([mes,MPI.FLOAT],source = source,tag = 10)
                total_error = total_error + mes[0]
            for dest in range(1,NP):
                mes = np.array([total_error],dtype='float')
                comm.Send([mes,MPI.FLOAT],dest = dest,tag = 11)
        if 1 <= rank < NP:
            mes = np.zeros(1,dtype='float')
            comm.Recv([mes,MPI.FLOAT],source = 0,tag = 11)
            total_error = mes[0]

def set_finalU():
    '''
    process 0 combine the subgrids from the processes 
    '''
    global finalU
    if rank < NP:
        if rank!=0:
            mes = local_U.copy()
            MPI.COMM_WORLD.Send([mes,MPI.FLOAT],dest = 0,tag = 12)
        if rank == 0:
            list_of_U = [0]*NP
            list_of_U[0] = local_U
            height = 4*N+1
            for source in range(1,NP):
                width = I[source]
                list_of_U[source] = np.zeros((height,width),dtype = 'float')
                MPI.COMM_WORLD.Recv([list_of_U[source],MPI.FLOAT],source = source, tag = 12)

            finalU = np.hstack((U[:,0].reshape(height,1),local_U))
            for j in range(1,len(list_of_U)):
                finalU = np.hstack((finalU,list_of_U[j]))
            finalU = np.hstack((finalU,U[:,6*N].reshape(height,1)))

def plot(finalU):
    xi = np.arange(0,3+h,h)
    yi = np.arange(0,2+h,h)
    plt.figure(1)
    plt.contour(xi,yi,finalU,10,linewidths=0.5,colors='k')
    plt.contourf(xi,yi,finalU,10,cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Stream function of the Irrotational flow")
    plt.xlim(0,3)
    plt.ylim(0,2)
    plt.show()

def initialise_U():
    '''
    initialise the grid U with the boundary conditions
    '''
    global U
    if rank == 0:
        # boundary conditions
        U = np.zeros((4*N+1,6*N+1),dtype ='float')
        for y in range(0,4*N+1): U[y][0] = 3*(y*h)          # left
        for y in range(2*N+1,3*N): U[y][6*N] = 12*(y*h-1)   # right
 
        for x in range(0,2*N+1): U[0][x] = 0                #|
        for y in range(1,2*N): U[y][2*N+y]= 0               #| bottom
        for x in range(4*N,6*N+1): U[2*N][x] = 0            #|

        for x in range(3*N,6*N+1): U[3*N][x] = 6            #|
        for y in range(3*N+1,4*N): U[y][2*N+(4*N-y)]= 6     #| top
        for x in range(0,2*N+1): U[4*N][x] = 6              #|

        # initial values of the internal nodes
        for x in range(1,6*N): 
            for y in range(1,4*N):
                if int_node(x,y): U[y][x] = 1

    # process 0 broadcasts the initial grid U 
    if rank == 0:
        for dest in range(1,NP):
            mes = U
            comm.Send([mes,MPI.FLOAT],dest=dest,tag=dest)
    if rank!= 0:
        mes = np.zeros((4*N+1,6*N+1),dtype='float')
        comm.Recv([mes,MPI.FLOAT],source = 0,tag = rank)
        U = mes

def grid_division():
    '''
    return the boundaries of the strips the width of them is variable:
    '''
    remaining_int_nodes = all_int_nodes()
    div = int(math.ceil(remaining_int_nodes/NP))
    M = [1] # the interval [0,3] will divided among the processors in intervals [M_p,M_p+1)
    for proc in range(NP):
        local_int_nodes = col_int_nodes(M[proc])
        width = 1
        while div > local_int_nodes: # if the division is < the internal nodes in the strip, we take that strip
            local_int_nodes = local_int_nodes + col_int_nodes(M[proc]+width)
            width = width + 1
        remaining_int_nodes = remaining_int_nodes - local_int_nodes 
        remaining_proc = NP - (proc + 1)
        M.append(M[proc] + width)
        if remaining_proc != 0: 
            div = int(math.ceil(remaining_int_nodes/remaining_proc)) # always divide the remaining internal nodes among the processors
        else: 
            M[proc+1] = 6*N
            break
    return M

# ========================================= main ===================================================
if __name__ == '__main__':
    status = MPI.Status()
    rank = comm.Get_rank()
    NP = comm.Get_size()

    h = 1./(2*N)             # thus we have 4N+1 rows and 6N+1 columns
    U = []                   # U will be modified by the function initialise_U
    U_right,U_left = [],[]   # U_right and U_left will be modified by the function boundary_exchange
    
    initialise_U()
    M = grid_division()

    x_int = [range(M[proc],M[proc+1]) for proc in range(len(M)-1)]  # store the interval [M_i,M_i+1)

    I = [len(interval) for interval in x_int]     # store the length of each interval [M_i,M_i+1)

    local_U = U[:,M[rank]:M[rank+1]]
    finalU = []              # finalU will be modified by the function set_finalU

    run()
    set_finalU()
    if rank == 0: plot(finalU)


