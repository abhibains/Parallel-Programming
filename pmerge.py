from mpi4py import MPI
import numpy as np
import time

# Initialize benchmark timer
start_time = time.time()

# Initialize arrays
sendbufA = []
sendbufB = []
sendbufBdemo = []

# mpi4py initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Array range and length
arrayrange = 10000000 
arraylength = 10000000

# Processor 0 creates random sorted arrays A & B
if rank == 0:
    arrA = np.random.randint(arrayrange, size=arraylength)
    arrA = np.sort(arrA)
    arrB = np.random.randint(arrayrange, size=arraylength)
    arrB = np.sort(arrB)

    # Initialize array to store last indices of each partition in A
    last_element_arr = []

    # Determining size of each partition of A
    nsize = arraylength / size
    
    # Loop to divide A into partitions
    for i in range(size):
        
        # Determining start & end indices of each partition in A
        start = i * (arraylength / size) + min(i, arraylength % size)
        start = int(start)
        end = start + nsize
        end = int(end)

        # Creating slices of A & storing their last indices
        sli_arrA= arrA[start:end]
        last= sli_arrA[-1]
        last_element_arr.append(last)

    # Initialize array to store last indices of each partition in B
    last_positionB=[]
    
    # Loop to divide B into partitions
    for element in last_element_arr:
        indexB = np.searchsorted(arrB,element)
        indexB=indexB-1
        last_positionB.append(indexB)
        
    # Creating array of arrays to store partitions of B
    position=0
    sli_arrB=[]
    for indices in last_positionB:
        sli_arrB=arrB[position:indices+1]
        if  indices == last_positionB[-1]:
            sli_arrB = arrB[position:]

        position=indices+1

        # sendBufBdemo stores partitions of B in a 2D array
        sendbufBdemo.append(sli_arrB)

    # sli_arrA stores partitions of A in a 2D array
    sli_arrA= np.array_split(arrA,size)

    # Initialising the  buffers that will be broadcasted to each processor
    sendbufA = sli_arrA
    sendbufB = sendbufBdemo

# mpi4py scatter command sends each partition automatically
# to the respective processor. Index i of sendbufA and sendbufB
# is sent to processor i
scatter_a = comm.scatter(sendbufA, root=0)
scatter_b = comm.scatter(sendbufB, root =0)

# Each processor i combines partitions from sendbufA and sendbufB
# and performs a sort operation
printerA = []
printerB = []
printerA = np.asarray(scatter_a).tolist()
printerB = np.asarray(scatter_b).tolist()
printerC = printerA + printerB
printerC.sort()

# Gather the output of all processors into a final array
finalized_arr = comm.gather(printerC, root=0)

if rank==0:
    C = np.array(finalized_arr, dtype=object)
    Cflat = np.hstack(C).tolist()
    # Benchmark prints the time it took to execute
    print("--- %s seconds ---" % (time.time() - start_time))
    sortedcheck = False
    for i in range(len(Cflat)-1):
        if Cflat[i] <= Cflat[i+1]:
            sortedcheck = True
            continue
        else:
            sortedcheck = False
            print("Cflat[i] = %i, Cflat[i+1] = %i." % (Cflat[i], Cflat[i+1]))
            break
    print("Array is sorted? %s" % sortedcheck)
