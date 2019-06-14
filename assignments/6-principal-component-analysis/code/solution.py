import numpy as np
'''
Homework6: Principal Component Analysis

Helper functions
----------------
In this assignment, you may want to use following helper functions:
- np.linalg.svd(): compute the singular value decomposition on a given matrix. 
- np.dot(): matrices multiplication operation.
- np.mean(): compute the mean value on a given matrix.
- np.ones(): generate a all '1' matrix with a certain shape.
- np.transpose(): matrix transpose operation.
- np.linalg.norm(): compute the norm value of a matrix. You may use it for the reconstruct_error function.

Principal component analysis function.

	Args:
	A: the data with shape (3000, 256). 3000 is the total number of samples and 256 is the total features/values of each sample.
	p: the number of principal components. A scatter number.

	Returns:
	U_p: 'p' principal components with shape (256, p).
A1: The reduced data matrix after PCA with shape (p, 3000).
'''
def pca(A, p):

    A = np.transpose(A)
    
    B = np.mean(A,1) * np.ones((1,256))
    B = np.transpose(B)
    B = B * np.ones((1,3000))
    A0 = A - B
    u, s, vh = np.linalg.svd(A0)
    
    u = u[:,0:p]
    A1 = np.dot(np.transpose(u), A)
    return u,A1
        
def reconstruction(U, A1):
    '''
	Reconstruct data function.

	Args:
	U: 'p' principal components with shape (256, p).
	A1: The reduced data matrix after PCA with shape (p, 3000).

	Return:
	Re_A: The reconstructed matrix with shape (3000, 256)
	'''
    Z = np.dot(U, A1)
    return np.transpose(Z)


def reconstruct_error(A, B):
    '''
	reconstruction error function.

	Args: 
	A & B: Two matrices needed to be compared with shape (3000, 256).

	Return: 
	error: the Frobenius norm's square of the matrix A-B. A scatter number.
	'''
    C = np.linalg.norm(A-B)
    return np.square(C)

