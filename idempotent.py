import numpy as np
import random as rand
from scipy.stats import unitary_group
from scipy.stats import ortho_group


np.printoptions(precision=3, suppress=True)

def purification(B):
    for i in range(4):
        B=3*np.matmul(B,B)-2*np.matmul(B,np.matmul(B,B))
    return B


def near_idempotent(N,size):
    B=np.zeros((size,size))
    i=0
    visited=[]
    while i<N:
        
        mixing=rand.random()*0.2
        temp_1=rand.randint(0,size-1)
        temp_2=rand.randint(0,size-1)
        if (temp_1 not in visited)and(temp_2 not in visited)and(temp_1!=temp_2):
            B[temp_1,temp_1]=1-mixing
            B[temp_2,temp_2]=0+mixing
            visited.append(temp_1)
            visited.append(temp_2)
            i+=1
    random_matrix=np.random.rand(size,size)
    eigenvalues,eigenvectors=np.linalg.eig(random_matrix)
    rum=ortho_group.rvs(size)

    with np.printoptions(precision=3, suppress=True):
        print(np.matmul(rum.conj().T,rum))
        print(np.trace(np.matmul(rum.conj().T,np.matmul(B,rum))))


    return B
N=6
size=2*N


A=np.identity(size)
B=near_idempotent(N,size)



print(np.trace(B-np.matmul(B,B)))

B_pure=purification(B)

print(np.trace(B_pure-np.matmul(B_pure,B_pure)))

