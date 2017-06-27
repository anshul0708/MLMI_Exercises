"""
    % This is the PCA implementation
    % This is the outline of your first excercise in Machine Learning for
    % Medical Application (MLMI) practical course
    % --------------------------------------------------------------------------------------------
    % Author
    % Shadi Albarqouni, PhD Candidate @ CAMP-TUM.
    % Conatct:               shadi.albarqouni@tum.de
    % --------------------------------------------------------------------------------------------
    % Copyright (c) 2016 TU Munich.
    % All rights reserved.
    % This work should be used for nonprofit purposes only.
    % --------------------------------------------------------------------------------------------
"""

import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# This function should implement the PCA using the Singular Value
#    % Decomposition (SVD) of the given dataMatrix
#        %
#        % Input:
#        %       dataMatrix (nrDims x nrSamples)
#        %       desiredVariancePercentage (%)
#        % Output is a structure
#        %       eigvecs: eigenvectors
#        %       eigvals: eigenvalues
#        %       meanDataMatrix
#        %       demeanedDataMatrix
#        %       projectedData

obj= {}

def usingSVD(dataMatrix, desiredVariancePercentage=1.0):
    # This function should implement the PCA using the Singular Value
    # Decomposition (SVD) of the given dataMatrix
    
    # De-Meaning the feature space
    obj['meanDataMatrix'] = np.mean(dataMatrix, axis=1)
    obj['demeanedDataMatrix'] = dataMatrix - obj['meanDataMatrix'][:, np.newaxis]
    
    # SVD Decomposition
    # You need to transpose the data matrix
    dataMatrix = dataMatrix.T
    U, sigma, V = np.linalg.svd(obj['demeanedDataMatrix'])
    # Enforce a sign convention on the coefficients -- the largest element (absolute) in each
    # column will have a positive sign.
    temp = np.argmax(np.abs(V), axis=0)
    for ind in range(len(temp)):
        V[temp[ind], ind] = np.abs(V[temp[ind], ind])
            
    # Compute the accumelative Eigenvalues to finde the desired
    # Variance
    eigenvals = sigma**2
    
    # Keep the eigenvectors and eigenvalues of the desired
    # variance, i.e. keep the first two eigenvectors and
    # eigenvalues if they have 90% of variance.
    total_diag = np.sum(eigenvals)
    k = 0
    total_var = 0
    for i in range(len(eigenvals)):
        k = i+1
        total_var += eigenvals[i]
        if total_var/total_diag >= desiredVariancePercentage:
            break

    obj['eigvecs'] = V[0:k]
    obj['eigvals'] = eigenvals[0:k]
    
    
    # Project the data
    dataReduced = np.dot(np.transpose(U)[0:k], dataMatrix.T)
    obj['projectedData'] = dataReduced
    
    # return the object
    return obj


# This function should implement the PCA using the EigenValue
#    % Decomposition of the given Covariance Matrix
#        %
#        % Input:
#        %       dataMatrix (nrDims x nrSamples)
#        %       desiredVariancePercentage (%)
#        % Output is a structure
#        %       eigvecs: eigenvectors
#        %       eigvals: eigenvalues
#        %       meanDataMatrix
#        %       demeanedDataMatrix
#        %       projectedData

 
def usingCOV(dataMatrix, desiredVariancePercentage=1.0):
    # This function should implement the PCA using the
    # EigenValue Decomposition of a given Covariance Matrix 
    
    # De-Meaning the feature space 
    obj['meanDataMatrix'] = np.mean(dataMatrix,axis=1)
    obj['demeanedDataMatrix'] = dataMatrix - obj['meanDataMatrix'][:, np.newaxis]
            
    # Computing the Covariance 
    obj['covMatrix'] = np.dot(obj['demeanedDataMatrix'], obj['demeanedDataMatrix'].T)

            
    # Eigen Value Decomposition
    
    eigval, eigvec = np.linalg.eigh(obj['covMatrix'])
    
    # In COV, you need to order the eigevectors according to largest eigenvalues
    
    idx = np.argsort(eigval)[::-1]
    eigvec = eigvec[:,idx]
    eigval = eigval[idx]
    
    
    # Enforce a sign convention on the coefficients -- the largest element (absolute) in each
    # column will have a positive sign.
    temp = np.argmax(np.abs(eigvec), axis=0)
    for ind in range(len(temp)):
        eigvec[temp[ind], ind] = np.abs(eigvec[temp[ind], ind])

    # Compute the accumelative Eigenvalues to finde the desired
    # Variance 
    
    total_sum = np.sum(eigval)
    k = 0
    total_var = 0
    for i in range(len(eigval)):
        k = i+1
        total_var += eigval[i]
        if total_var/total_sum >= desiredVariancePercentage:
            break
    eigvec = eigvec[:, :k]
    # Keep the eigenvectors and eigenvalues of the desired
    # variance, i.e. keep the first two eigenvectors and
    # eigenvalues if they have 90% of variance. 
    obj['eigvecs'] = eigvec
    obj['eigvals'] = eigval[0:k]
            
    # Project the data 
    obj['projectedData'] = np.dot(np.transpose(eigvec), dataMatrix)
    
    # return the object
    return obj

mat = sio.loadmat('/home/nomad/Documents/Projects/MLMI/05_exercise/Assignment_PythonKmeansPCA/Assignment_Python/data/filtHeartDataSet.mat')
dataMat = mat['dataMatrix']
labels = mat['labels']
print labels.shape
ans =  usingSVD(dataMat.T, 0.98)
print ans['projectedData'].shape


COLORS = np.array(['r' if label == 'Yes' else 'b' for label in labels])


ind = np.where(COLORS=='r')
data_r = ans['projectedData'][:, ind]
data_r = np.reshape(data_r,(data_r.shape[0],data_r.shape[2]))
# data_b = ans['projectedData'][:,np.where(COLORS=='b')]
ind = np.where(COLORS=='b')
data_b = ans['projectedData'][:, ind]
data_b = np.reshape(data_b,(data_b.shape[0],data_b.shape[2]))


        
data_r_mean = np.mean(data_r, axis=1)
data_b_mean = np.mean(data_b, axis=1)

sse_yes =  np.sqrt(((data_r - data_r_mean[:,np.newaxis])**2).sum(axis = 0)).sum()
sse_no = np.sqrt(((data_b - data_b_mean[:,np.newaxis])**2).sum(axis = 0)).sum()
print " With Heart Disease SSE : %f" % sse_yes
print " Without Heart Disease SSE : %f" % sse_no

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ans['projectedData'][0], ans['projectedData'][1], ans['projectedData'][2], c=COLORS)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
