import numpy as np
from scipy.stats import ortho_group


np.printoptions(precision=3, suppress=True)

def purification(B):
    for i in range(5):
        B=3*np.matmul(B,B)-2*np.matmul(B,np.matmul(B,B))
    return B


def near_idempotent(N,size,mixing_val=0.2):
    matrix=np.zeros((size,size))
    i=0
    visited=[]
    while i<N:
        mixing=np.random.random()*mixing_val
        temp_1=np.random.randint(0,size)
        temp_2=np.random.randint(0,size)
        if (temp_1 not in visited)and(temp_2 not in visited)and(temp_1!=temp_2):
            matrix[temp_1,temp_1]=1-mixing
            matrix[temp_2,temp_2]=0+mixing
            visited.append(temp_1)
            visited.append(temp_2)
            i+=1

    rum=ortho_group.rvs(size)

    return matrix, np.matmul(rum.conj().T,np.matmul(matrix,rum))

N=5
size=2*N


P_1_diag,P_1=near_idempotent(N,size)
P_2_diag,P_2=near_idempotent(N,size)
P_1_pure=purification(P_1)
with np.printoptions(precision=3, suppress=True):
    print(P_1)
    print(P_1_pure)
print("Trace of not purifie matrix %3.3f"%np.trace(P_1))
print("Trace purifie matrix %3.3f"%np.trace(P_1_pure))

print("Trace of density matrix difference %3.3f"%np.trace(P_1-P_2))

eigenvals,eigenvectors=np.linalg.eig(P_1)


with np.printoptions(precision=3, suppress=True):
    print(np.matmul(eigenvectors.conj().T,np.matmul(P_1,eigenvectors)).diagonal())
    print(P_1_diag.diagonal())


