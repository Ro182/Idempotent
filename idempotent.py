import numpy as np
from scipy.stats import ortho_group
from scipy.linalg import lapack
import scipy as scp

np.set_printoptions(precision=3, suppress=True)

def unitary_matrix(size):
    matrix=np.zeros((size,size),dtype=np.float64)
    for i in range(size-1):
        for j in range(i+1,size):
            matrix[i][j]=matrix[j][i]=np.random.rand()
    return matrix




def purification(matrix,max_iterations=5 ,precision=1e-6):
    """This function takes a almost idempotent matrix P and through McWeeny purification, 
    creates a idempotent matrix P'. It uses the next mapping:
    P'=3PP-2PP
    The mapping can be done many untils until achieve a desire precision.
    If the precision is not achieved until a max_number of it will print out a warning and  the precision achieved
    The precision is calculated for the matrix element with the maximum absolute value. 
    """
    max_val=np.max(np.abs(np.matmul(matrix,matrix )-matrix))
    counter=1
    while (max_val>precision):
        matrix=3*np.matmul(matrix,matrix,dtype=np.float64)-2*np.matmul(matrix,np.matmul(matrix,matrix,dtype=np.float64),dtype=np.float64)
        counter+=1
        max_val=np.max(np.abs(np.matmul(matrix,matrix )-matrix))
        if counter> max_iterations:
            print("Maximun number of iteractions achieve before convergances")
            print("Precision achieve = %e"%max_val)
            return matrix
    print("Precision achieve = %2.2e in %i iterations"%(precision,counter))
    return matrix


def near_idempotent(N,size,mixing_val=0.2):
    """The function returns a near idempotent matrix.
    For this it creates a matrix of a given size, a fill the diagonal of this matrix
    with N par values, such as that each pair add up to 1, and the values are positive, 
    so the trace of this matrix is equal to N, e.i. Tr(matrix)=N
    
    Then it does an unitary transformation of the matrix so it is not longer diagonal.
    The variavle mixing, allows to control how far away from 0 and 1 each value can diverge. 
    """
    matrix=np.zeros((size,size),dtype=np.float64)
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
    #rum=unitary_matrix(size)
    #rum=np.identity(size)
    print("Max absolute element of P^2-P:",np.max(np.abs(np.matmul(matrix,matrix )-matrix)))
    non_diag_matrix=np.matmul(rum.conj().T,np.matmul(matrix,rum ) )
    #print("Matrix rank of the unitary transformation=",np.linalg.matrix_rank(rum))
    print("Trace of almost idempotent matrix=",np.trace(non_diag_matrix))
    return matrix, non_diag_matrix



N=8
size=2*N

print("Size of the matrix: %i x %i, Number of electrons:%i"%(size,size,N))

P_1_diag,P_1=near_idempotent(N,size)

print("Almost idempotent matrix:\n",P_1)

print("Matrix purification")
P_1_pure=purification(P_1)

print("Purified matrix:\n",P_1_pure)
print("Diference between the matrices\n",P_1-P_1_pure)
print("Maximun absolute difference: %f"%np.max(abs(P_1-P_1_pure)))

print("Trace of almost idempotent matrix %f"%np.trace(P_1))
print("Trace purifie matrix %f"%np.trace(P_1_pure))


#eigenvals,eigenvals_i,eigenvectors,eigenvectors_i,_=lapack.dgeev(P_1_pure)

#eigenvals,eigenvectors=np.linalg.eig(P_1_pure)
eigenvals,eigenvectors=scp.linalg.eig(P_1_pure)


P_1_in_new_base=np.matmul(eigenvectors.conj().T,np.matmul(P_1,eigenvectors,dtype=np.float64),dtype=np.float64)


    
print("Eigenvalues of the almost idempotent matrix in the purified matrix based and trace:\n",np.sort(P_1_in_new_base.diagonal()),np.trace(P_1_in_new_base))
print("Eigenvalues of the almost idempotent matrix and trace:\n",np.sort(P_1_diag.diagonal()),np.trace(P_1_diag))

print("Almost idempotent matrix in the purified matrix basis set:")
print(P_1_in_new_base)

print("Diagonalized Purified matrix:")
print(np.matmul(eigenvectors.conj().T,np.matmul(P_1_pure,eigenvectors,dtype=np.float64),dtype=np.float64))






