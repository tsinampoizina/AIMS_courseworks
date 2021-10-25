# Game of life
# Randriamahefasoa Tsinampoizina Marie-Sophie

import numpy.random
import numpy as np
import math
import mpi4py.MPI as MPI
import sys
from time import sleep
from pylab import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt
comm = MPI.COMM_WORLD
def process_coordinates(rank):
    '''
    return the coordinates (p,q) of a processor given its rank
    '''
    p = rank % P
    q = int((rank - p)/P)
    return (p,q)

def process_rank(p,q): 
    '''
    return the rank of the processor with reference to his coordinates
    '''
    return int(q*P+p)
    
def global_indices(j,i):
    '''
    return the global indices (of matrix subA) which correspond to the local indices (AA)
    '''
    return q+j,p+i

def next_state(Sji,j,i):
    '''
    modify the state of the cell according to its neighbors following the rule
    '''
    global subA
    if subA[j,i] == 1 and (Sji<2 or Sji>3): subA[j,i] = -1
    if subA[j,i]==0 and Sji == 3: subA[j,i] = 2

def f(x):
    '''
    this function satisfies f(1)=f(-1)=1 and f(0)=f(2)=0
    '''
    return abs(x)*(2-abs(x))

def sum_f(a1,a2,a3,a4,a5,a6,a7,a8):
    '''
    sum f(x) for the 8 neighbors x=a,b,...,h
    '''
    ls = [f(a1),f(a2),f(a3),f(a4),f(a5),f(a6),f(a7),f(a8)]
    return int(sum(ls))

def next_grid():
    '''
    change the values of all cells in the submatrix subA of a process
    '''
    global subA

    # i == 0 and j == 1,...,J-2
    i = 0
    for j in range(1,J[q]-1):
        Sji = sum_f(subAleft[j],subAleft[j-1],subAleft[j+1],subA[j,i+1],subA[j-1,i+1],subA[j+1,i+1],subA[j-1,i],subA[j+1,i])
        next_state(Sji,j,i)

    # i == 1,...,I-2 and j == 1,...,J-2
    for i in range(1,I[p]-1):
        for j in range(1,J[q]-1):
            Sji = sum_f(subA[j,i-1],subA[j-1,i-1],subA[j+1,i-1],subA[j,i+1],subA[j-1,i+1],subA[j+1,i+1],subA[j-1,i],subA[j+1,i])
            next_state(Sji,j,i)

    # i == I-1 and j == 1,...,J-2
    i = I[p]-1
    for j in range(1,J[q]-1):
        Sji=sum_f( subA[j,i-1],subA[j-1,i-1],subA[j+1,i-1],subAright[j],subAright[j-1],subAright[j+1],subA[j-1,i],subA[j+1,i])
        next_state(Sji,j,i)

    # j == 0 and i == 1,...,I-2
    j = 0
    for i in range(1,I[p]-1):
        Sji = sum_f(subA[j,i-1],subAbottom[i-1],subA[j+1,i-1],subA[j,i+1],subAbottom[i+1],subA[j+1,i+1],subAbottom[i],subA[j+1,i])
        next_state(Sji,j,i)

    # j == Jq-1 and i == 1,...,I-2
    j = J[q]-1
    for i in range(1,I[p]-1):
        Sji = sum_f(subA[j,i-1],subA[j-1,i-1],subAtop[i-1],subA[j,i+1],subA[j-1,i+1],subAtop[i+1],subA[j-1,i],subAtop[i])
        next_state(Sji,j,i)

    # 4 corners
    # bottomleft subA[0,0]
    j,i = 0,0
    Sji = sum_f(subAleft[j] ,  subAbottomLeft[0]  ,subAleft[j+1],subA[j,i+1], subAbottom[i+1],subA[j+1,i+1], subAbottom[i] ,subA[j+1,i])
    next_state(Sji,j,i)
            
    # bottomright subA[0,Ip-1]
    j,i = 0,I[p]-1
    Sji = sum_f(subA[j,i-1],subAbottom[i-1],subA[j+1,i-1],subAright[j],subAbottomRight[0],subAright[j+1],subAbottom[i],subA[j+1,i])
    next_state(Sji,j,i)
    
    # topleft subA[Jq-1,0]
    j,i = J[q]-1,0
    Sji = sum_f(subAleft[j],subAleft[j-1],subAtopLeft[0],subA[j,i+1],subA[j-1,i+1],subAtop[i+1],subA[j-1,i],subAtop[i])
    next_state(Sji,j,i)
    
    # topright subA[Jq-1,Ip-1]
    j,i = J[q]-1,I[p]-1
    Sji = subA[j,i-1],subA[j-1,i-1],subAtop[i-1],subAright[j],subAright[j-1],subAtopRight[0],subA[j-1,i],subAtop[i]
    next_state(Sji,j,i)
    
    for m in range(0,J[q]):
        for n in range(0,I[p]): 
            if subA[m,n] == 2: subA[m,n] = 1      # now change the 2 to 0
            if subA[m,n] == -1: subA[m,n] = 0      # now change the -1 to 1

    return 0

    
def boundary_exchange(subA,I,J):
    global subAleft,subAright,subAbottom,subAtop,subAbottomLeft,subAbottomRight,subAtopLeft,subAtopRight

############################################
    # send the top
    if 0 <= q < Q-1:
        mes1 = subA[J[q]-1,0:I[p]]
        comm.Send( [mes1,MPI.INT] , dest= process_rank(p,q+1) , tag=1 )
    if q == Q-1 :   # toroidal
        mes1 = subA[J[q]-1,0:I[p]]
        comm.Send( [mes1,MPI.INT], dest= process_rank(p,0) , tag= 1 )

    # receive the bottom
    if 1 <= q < Q:
        mes11 = np.zeros(I[p],dtype='int')
        comm.Recv( [mes11,MPI.INT], source= process_rank(p,q-1) , tag= 1 )
        subAbottom = mes11
    if q == 0:
        mes11 = np.zeros(I[p],dtype='int')
        comm.Recv( [mes11,MPI.INT], source= process_rank(p,Q-1) , tag= 1)
        subAbottom = mes11

############################################
    # send bottom 
    if 1 <= q < Q:
        mes2 = subA[0,0:I[p]]
        comm.Send( [mes2,MPI.INT], dest= process_rank(p,q-1) , tag= 2 )

    if q == 0:
        mes2 = subA[0,0:I[p]]
        comm.Send( [mes2,MPI.INT], dest= process_rank(p,Q-1)  , tag= 2 )
    
    # receive top
    if 0 <= q < Q-1:
        mes22 = np.zeros(I[p],dtype='int')
        comm.Recv( [mes22,MPI.INT], source= process_rank(p,q+1) , tag = 2)
        subAtop = mes22
    if q == Q-1:
        mes22 = np.zeros(I[p],dtype='int')
        comm.Recv( [mes22,MPI.INT], source= process_rank(p,0) , tag = 2)
        subAtop = mes22

############################################
    # send right
    if 0 <= p < P-1:
        mes3o = subA[0:J[q],I[p]-1]
        mes3 = mes3o.copy()
        comm.Send( [mes3,MPI.INT], dest= process_rank(p+1,q) , tag = 3 )
    if p == P-1:
        mes3o = subA[0:J[q],I[p]-1]
        mes3 = mes3o.copy()
        comm.Send( [mes3,MPI.INT], dest = process_rank(0,q) , tag= 3 )
    # receive left
    if 1 <= p < Q:
        mes33 = np.zeros(J[q],dtype='int')
        comm.Recv( [mes33,MPI.INT], source= process_rank(p-1,q)  , tag = 3)
        subAleft = mes33
    if p == 0:
        mes33 = np.zeros(J[q],dtype='int')
        comm.Recv( [mes33,MPI.INT], source = process_rank(P-1,q) , tag = 3)
        subAleft = mes33

############################################
    # send left
    if 1 <= p < Q:
        mes4o =  subA[0:J[q],0]
        mes4 = mes4o.copy()
        comm.Send( [mes4,MPI.INT], dest = process_rank(p-1,q) , tag = 4)
    if p == 0:
        mes4o = subA[0:J[q],0]
        mes4 = mes4o.copy()
        comm.Send( [mes4,MPI.INT], dest = process_rank(P-1,q) , tag = 4)
     
    # receive right
    if 0 <= p < P-1:
        mes44 = np.zeros(J[q],dtype='int')
        comm.Recv( [mes44,MPI.INT], source = process_rank(p+1,q)  , tag = 4)
        subAright = mes44
    if p == P-1:
        mes44 = np.zeros(J[q],dtype='int')
        comm.Recv( [mes44,MPI.INT], source  = process_rank(0,q), tag  = 4)
        subAright = mes44

################  4 corners ################### 
###############################################
    #####     bottom_left   #################
    mes5 = np.array([subA[0,0]],dtype='int')
    if p == 0 and q == 0 : dest = process_rank(P-1,Q-1)
    elif p == 0 and 1 <= q <= Q-1 : dest = process_rank(P-1,q-1)
    elif 1 <= p <= Q-1 and q == 0 : dest = process_rank(p-1,Q-1)
    else:dest = process_rank(p-1,q-1)
    tag = 5
    comm.Send( [mes5,MPI.INT], dest , tag )

    # receive top - right
    mes55 =  np.zeros(1,dtype = 'int')
    if p == P-1 and q == Q-1: source = process_rank(0,0)
    elif 0 <= p <= P-2 and q == Q-1: source = process_rank(p+1,0)
    elif p == P-1 and 0 <= q <= Q-2: source = process_rank(0,q+1) 
    else:source = process_rank(p+1,q+1)
    tag = 5
    comm.Recv( [mes55,MPI.INT], source , tag , status)
    subAtopRight = mes55



############################################
    # send bottom - right
    mes6 = np.array([subA[0,I[p]-1]],dtype='int')
    if p == P-1 and q == 0: dest = process_rank(0,Q-1)
    elif p == P-1 and 1 <= q <= Q-1: dest = process_rank(0,q-1)
    elif 0 <= p <= P-2 and q == 0: dest = process_rank(p+1,Q-1)
    else:dest = process_rank(p+1,q-1)
    tag = 6
    comm.Send( [mes6,MPI.INT], dest , tag ) 
    
    # receive top - left
    mes66 = np.zeros(1,dtype='int')
    if p == 0 and q == Q-1 : dest = process_rank(P-1,0)
    elif p == 0 and 0 <= q <= Q-2 : dest = process_rank(P-1,q+1)
    elif 1 <= p <= P-1 and q == Q-1 : dest = process_rank(p-1,0)
    else:source = process_rank(p-1,q+1)
    tag = 6
    comm.Recv( [mes66,MPI.INT], source , tag , status)
    subAtopLeft = mes66


############################################
    # send top - left
    mes7 = np.array([subA[J[q]-1,0]],dtype = 'int')
    if p == 0 and q == Q-1 : dest = process_rank(P-1,0)
    elif p == 0 and 0 <= q <= Q-2 : dest = process_rank(P-1,q+1)
    elif 1 <= p <= P-1 and q == Q-1 : dest = process_rank(p-1,0)
    else:dest = process_rank(p-1,q+1)
    tag = 7
    comm.Send( [mes7,MPI.INT], dest , tag)

    # receive bottom - right
    mes77 = np.zeros(1,dtype='int')
    if p == P-1 and q == 0: source = process_rank(0,Q-1)
    elif p == P-1 and 1 <= q <= Q-1: source = process_rank(0,q-1)
    elif 0 <= p <= P-2 and q == 0: source = process_rank(p+1,Q-1)
    else: source = process_rank(p+1,q-1)
    tag = 7
    comm.Recv( [mes77,MPI.INT], source , tag , status) 
    subAbottomRight = mes77


############################################
    # send top - right
    mes8 = np.array([subA[J[q]-1,I[p]-1]],dtype = 'int')
    if p == P-1 and q == Q-1: dest = process_rank(0,0)
    elif 0 <= p <= P-2 and q == Q-1: dest = process_rank(p+1,0)
    elif p == P-1 and 0 <= q <= Q-2: dest = process_rank(0,q+1) 
    else: dest = process_rank(p+1,q+1)
    tag = 8
    comm.Send( [mes8,MPI.INT], dest , tag )
    
    # receive bottom - left
    mes88 = np.zeros(1,dtype = 'int')
    if p == 0 and q == 0 : source = process_rank(P-1,Q-1)
    elif p == 0 and 1 <= q <= Q-1 : source = process_rank(P-1,q-1)
    elif 1 <= p <= Q-1 and q == 0 : source = process_rank(p-1,Q-1)
    else:source = process_rank(p-1,q-1)
    tag = 8
    comm.Recv( [mes88,MPI.INT], source , tag , status)
    subAbottomLeft = mes88


if __name__ == '__main__':
    status = MPI.Status()
    rank = comm.Get_rank()
    NP = comm.Get_size()
    M = 12
    N = 12
    if rank == 0:
        print 'You need to enter the value of P,Q such that np = P*Q'
        P = int(raw_input())
        Q = int(raw_input())
        for dest in range(1,NP):
            tag = 0
            mes  = np.array([P,Q],dtype='int')
            comm.Send( [mes ,MPI.INT], dest , tag)
    if rank != 0:
        tag = 0
        mes = np.zeros(2,dtype='int')
        source = 0
        comm.Recv( [mes,MPI.INT], source , tag , status)
        P,Q = mes[0],mes[1]
    Mp = range(0,M+int(M/P),int(M/P))
    Nq = range(0,N+int(N/Q),int(N/Q))
    I = [Mp[x+1]-Mp[x] for x in range(0,len(Mp)-1)]
    J = [Nq[x+1]-Nq[x] for x in range(0,len(Nq)-1)]
    size = N*M
    AA = [np.random.randint(0,2) for i in range(0,size)]
    AA = np.array(AA,dtype='int').reshape(N,M)
    p,q = process_coordinates(rank)
    ql=Nq[q]
    qu=Nq[q+1]
    pl=Mp[p]
    pu=Mp[p+1]
    subA = AA[ql:qu,pl:pu]
    subAleft,subAright,subAbottom,subAtop,subAbottomLeft,subAbottomRight,subAtopLeft,subAtopRight = [],[],[],[],[],[],[],[]
    if rank == 0:
        plt.ion()
        plt.hold(False)
        fig1 = plt.figure(1)
        ax1 = fig1.gca()
    steps = 10
    for i in range(steps):
        boundary_exchange(subA,I,J)
        next_grid()
        # each process send the matrix subA to 0, but before they have to send its shape
        if rank != 0:
            mes = np.array(shape(subA),dtype='int')
            dest = 0
            tag = 9
            comm.Send([mes,MPI.INT],dest,tag)
        if rank == 0:
            shapes = [0]*NP
            tag = 9
            for source in range(1,NP):
                shapes[source] = np.zeros(2,dtype='int')
                comm.Recv([shapes[source],MPI.INT],source,tag,status)
        # each process send the matrix subA to 0
        if rank!=0:
            mes = subA.copy()
            dest = 0
            tag = 10
            comm.Send([mes,MPI.INT],dest,tag)
        if rank == 0:
            listA = [0]*NP
            tag = 10
            listA[0] = subA
            for source in range(1,NP):
                w,h = shapes[source][0],shapes[source][1]
                listA[source] = np.zeros(w*h,dtype = 'int').reshape(w,h)
                comm.Recv([listA[source],MPI.INT],source, tag, status)  
        # build the big matrix AA from the subMatrices subA
            rows = [0]*Q
            for qq in range(0,Q):
                rows[qq] = listA[qq*P]
                for pp in range(1,P):
                    rows[qq] = np.hstack((rows[qq],listA[pp+P*qq]))
            
            AA = rows[0]
            for ii in range(1,len(rows)):
                AA = np.vstack((AA,rows[ii]))
                
            plt.imshow(AA, interpolation='nearest', axes=ax1, animated=True, cmap=cm.gray)
            plt.draw()
            print AA
            sleep(0.1)
