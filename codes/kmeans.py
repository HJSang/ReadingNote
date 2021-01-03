# imports
import numpy as np
from numpy.linalg import norm
# KMeans class
class KMeans:
  def __init__(self,K,max_iter=100,random_state=2021):
    self.K = K
    self.max_iter = max_iter
    self.random_state = random_state
    self.sse = 0
  def init_centroids(self,X):
    # set random seed to make it reproducible 
    np.random.seed(self.random_state)
    # Randm shuffle the index
    idx = np.random.permutation(X.shape[0])
    # Initial the centroids as the first K idx 
    centroids = X[idx[:self.K]]
    return centroids
  def calculate_centroids(self, X, labels):
    # Initial centroids as 0 matrix
    centroids = np.zeros((self.K,X.shape[1]))
    for i in range(self.K):
      # average the samples from i-th label
      # axis=0: means the average operator is done by the row direction
      centroids[i,:] = np.mean(X[labels==i,:], axis=0)
    return centroids
  def calculate_distance(self, X, centroids):
    # distance[i,j] is the distance between sample i and centroids j.
    distance = np.zeros((X.shape[0], self.K))
    for i in range(self.K):
      # axis = 1: means the norm operation is done by the column direction
      distance[:,i]=norm(X - centroids[i,:], axis=1)
    return distance
  def find_label(self, distance):
    # distance[i,j] is the distance between sample i and centroids j.
    # axis=1: means find the index of minimum for each row.
    return np.argmin(distance, axis=1)
  def calculate_sse(self, X, labels, centroids):
    dist = np.zeros(X.shape[0])
    for i in range(self.K):
      dist[labels==i] = norm(X[labels==i,:] - centroids[i],axis=1)
    return np.sum(np.square(dist))
  def fit(self,X):
    # Initialize
    self.centroids = self.init_centroids(X)
    for iter in range(self.max_iter):
      old_centroids = self.centroids
      distance = self.calculate_distance(X,self.centroids)
      labels = self.find_label(distance)
      self.centroids = self.calculate_centroids(X,labels)
      if np.all(self.centroids==old_centroids):
        break
    self.sse = self.calculate_sse(X,labels,self.centroids)

  def predict(self,X):
    distance = self.calculate_distance(X,self.centroids)
    return self.find_label(distance)

