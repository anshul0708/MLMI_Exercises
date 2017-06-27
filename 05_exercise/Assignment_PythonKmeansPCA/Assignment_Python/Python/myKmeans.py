"""
    % K-Means Implementation
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
#   % This function should assign the points in your dataMatrix to the
#    % closest centroid, the distance is computed using the L2 norm
#        %
#        % Input:
#        %       dataMatrix (nrDims x nrSamples)
#        %       centroids (nrDims x nrCentroids)
#        % Output
#        %       assigedPoints (1 x nrSamples) should have the centroid's
#        %       index

def assignPoints(dataMatrix, centroids):
    
    # return the assigned points
    dataMatrix = dataMatrix.T
    centroids = centroids.T
    distances = np.sqrt(((dataMatrix - centroids[:, np.newaxis])**2).sum(axis=2))
    assignedPoints = np.argmin(distances, axis=0)

    return assignedPoints




#    % This function should assign the points in your dataMatrix to the
#    % closest centroid, the distance is computed using the L2 norm
#        %
#        % Input:
#        %       dataMatrix (nrDims x nrSamples)
#        %       assigedPoints (1 x nrSamples)
#        % Output
#        %       updatedCentroids (nrDims x nrCentroids)


def updateCentroids(dataMatrix, assignedPoints,centroids):

    dataMatrix = dataMatrix.T

    updatedCentroids = np.array([dataMatrix[assignedPoints==k].mean(axis=0) for k in range(centroids.shape[1])])
    # return the updated centroids
    return updatedCentroids.T




#   % This function should compute the cost function
#    %
#        % Input:
#        %       dataMatrix (nrDims x nrSamples)
#        %       centroids (nrDims x nrCentroids)
#        %       assignedPoints (1 x nrSamples)
#        % Output
#        %       cost


def computeCost(dataMatrix, centroids, assignedPoints):

    dataMatrix = dataMatrix.T
    cost = 0
    centroids = centroids.T
    for k in range(centroids.shape[0]):
        cost += np.sqrt(((dataMatrix[assignedPoints==k] - centroids[k])**2).sum(axis = 1)).sum()
 
    return cost




#    % This function should run the K-Means algorithm
#    %
#        % Input:
#        %       dataMatrix (nrDims x nrSamples)
#        %       numberOfCluster
#        %       numberOfRuns
#        %       Tol
#        % Output
#        %       object


def runKmeans(dataMatrix, numberOfCluster, numberOfRuns=1, Tol=1e-6):
    
#    % You can neglect the number of runs at the begining, once you
#    % are done, you can work on it. The idea is to run the Kmeans
#    % several times (runs), then choose the one giving you the min.
#    % cost.
            
            
#    % ....
#    Note that you need to change any necessary syntax for python

    centroids = np.array(([]))
    assignedPoints = np.array([])
    tot_cost = []
    cents ={}
    asps ={}
    for i in range(numberOfRuns):
        
        centroids = dataMatrix[:,np.random.choice(np.arange(dataMatrix.shape[1]), numberOfCluster)]
        RE = 1
        cost = 0
        ind = 0
        prev_cost = 0
        while abs(RE)>=Tol:
            assignedPoints = assignPoints(dataMatrix, centroids)
            centroids = updateCentroids(dataMatrix, assignedPoints, centroids)
            
            cost = computeCost(dataMatrix, centroids, assignedPoints)

            if ind > 0:
                RE = (cost - prev_cost)/max(cost, prev_cost)
            prev_cost = cost
            ind += 1
        tot_cost.append(cost)
        cents.update({i:centroids})
        asps.update({i: assignedPoints})

            # for i = 1:numberOfRuns
            #     % initializa Centroids
            #     rndCen = randperm(nrSamples);
            #     centroids = dataMatrix(:,rndCen(1:objKmeans.numberOfCluster));
                
            #     if nargin <=3
            #         Tol = 1e-6;
            #     end
                
            #     iter = 0; RE = 1;
                
            #     while abs(RE) >= Tol % convergance criterion
            #         iter = iter + 1;
                    
            #         % .....
                    
                    
            #         cost(iter) =
                    
            #         % check the convergance
            #         if iter == 1;
            #             RE = 1;
            #         else
            #             RE = (cost(iter) - cost(iter-1))/max(cost(iter-1), cost(iter));
            #         end
                
            #     end
                
                
            #     % ......
        
            # end
        
    min_ind = np.argmin(tot_cost)
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dataMatrix[0],dataMatrix[1], dataMatrix[2], c=asps[min_ind])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    return cents[min_ind], tot_cost[min_ind], asps[min_ind]


#        % This function should run the K-Means algorithm for visualization
#        % Input:
#        %       dataMatrix (nrDims x nrSamples)
#        %       numberOfCluster
#        % Output
#        %       object
        
def runKmeansVis(dataMatrix, numberOfCluster):
    centroids = dataMatrix[:,np.random.choice(np.arange(dataMatrix.shape[1]), numberOfCluster)]
    RE = 1
    cost = 0
    ind = 0
    prev_cost = 0
    Tol = 1e-6
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()
    while abs(RE)>=Tol:
        assignedPoints = assignPoints(dataMatrix, centroids)
        centroids = updateCentroids(dataMatrix, assignedPoints, centroids)
        
        cost = computeCost(dataMatrix, centroids, assignedPoints)

        if ind > 0:
            RE = (cost - prev_cost)/max(cost, prev_cost)
        
        
        ax.scatter(dataMatrix[0],dataMatrix[1], dataMatrix[2], c=assignedPoints)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.pause(0.05)
        ind += 1
    
        

# return objKmeansVis

mat = sio.loadmat('/home/nomad/Documents/Projects/MLMI/05_exercise/Assignment_PythonKmeansPCA/Assignment_Python/data/toydata2.mat')
dataMat = mat['data']
runKmeans(dataMat, 8, 10)

