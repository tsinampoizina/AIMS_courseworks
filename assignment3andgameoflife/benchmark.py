# Ping-Pong Benchmark
# Randriamahefasoa Tsinampoizina Marie-Sophie


import numpy as np
import mpi4py.MPI as MPI
import time


if __name__ == '__main__':
    status = MPI.Status()
    p = MPI.COMM_WORLD.Get_rank()
    NP = MPI.COMM_WORLD.Get_size()
    start = time.time()
    
    for k in range(0,19,2): # I could not find difference according to the size so I take a bigger range of size [0,2,...20]
        N = 2**k
        x = np.arange(1,N+1)
        
        for i in range(0,17):
            if p%2 == 0 and p!= NP-1:   # even compute node and not the last one
                start = time.time()
                tag = 1
                dest = p+1
                MPI.COMM_WORLD.Send([x,MPI.INT],dest,tag)

                source = p+1
                tag = 2
                x = np.zeros(N,dtype = 'int')
                MPI.COMM_WORLD.Recv([x,MPI.INT],source ,tag,status)
                if i == 16 and p == 0: print 'k = %d: Between Process'%k, p,'and %d: Running time:'%(p+1),time.time()-start  # I only print the time between process 0 and 1 which I think is enough since we repeat 5 times

                
            elif p%2 == 1:
                source = p-1
                tag = 1
                x = np.zeros(N,dtype = 'int')
                MPI.COMM_WORLD.Recv([x,MPI.INT],source ,tag,status)
            
                dest = p-1
                tag = 2
                MPI.COMM_WORLD.Send([x,MPI.INT],dest,tag)
            

    

