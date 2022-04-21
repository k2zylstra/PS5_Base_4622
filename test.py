#!pip install -r requirements.txt #uncomment to install all required packages
import numpy as np
import matplotlib.pyplot as plt
import tests
import data
from sklearn.svm import SVC


circles = data.Circles()
multi_blobs = data.DataBlobs(centers=5, std=1.5)
binary_blobs = data.DataBlobs(centers=2, std=2.5)

fig, axs = plt.subplots(1, 3)
fig.set_figheight(4), fig.set_figwidth(12)
for i, (dataset, name) in enumerate([(circles, "Co-centric circles"),
                                     (multi_blobs, "multi-blobs"),
                                     (binary_blobs, "binary blobs")]):
    axs[i].set_title(name)
    axs[i].scatter(dataset.X[:, 0], dataset.X[:, 1], c=dataset.labels)
#plt.show()

class KMeans:
    def __init__(self, k, rtol=1e-3):
        """
        :param k: (int) number of means/centroids to evaluate
        :param rtol: (float) relative tolerance, `epsilon` from the markdown
        """
        self.k = k
        self.centroids = None
        self.snapshots = []  # buffer for progress plots
        self.rtol = rtol

    def initialize_centroids(self, X):
        """
        Randomly select k **distinct** samples from the dataset in X as centroids
        @param X: np.ndarray of dimension (num_samples, num_features)
        @return: centroids array of shape (k, num_features)
        """
        centroids = None
        # Workspace 1.1
        #BEGIN 
        centroids = []
        a = [i for i in range(X.shape[0])]
        c = np.random.choice(a, size=self.k, replace=False)
        for i in c:
            centroids.append(X[i])
                
        #TODO may need to assign to self.centroids
        #END
        return centroids

    def compute_distances(self, X):
        """
        Compute a distance matrix of size (num_samples, k) where each cell (i, j) represents the distance between
        i-th sample and j-th centroid. We shall use Euclidean distance here.
        :param X: np.ndarray of shape (num_samples, num_features)
        :return: distances_matrix : (np.ndarray) of the dimension (num_samples, k)
        """
        distances_matrix = np.zeros((X.shape[0], self.k))
        # Workspace 1.2
        #BEGIN 
        for i in range(X.shape[0]):
            for k in range(self.k):
                distances_matrix[i][k] = np.linalg.norm(X[i]-self.centroids[k])
                    
        #END
        return distances_matrix

    @staticmethod
    def compute_assignments(distances_to_centroids):
        """
        Compute the assignment array of shape (num_samples,) where assignment[i] = j if and only if
        sample i belongs to the cluster of centroid j
        :param distances_to_centroids: The computed pairwise distances matrix of shape (num_samples, k)
        :return: assignments array of shape (num_samples,)
        """

        assignments = np.zeros((distances_to_centroids.shape[0],), dtype=int)

        # Workspace 1.3
        #BEGIN 
        for i in range(distances_to_centroids.shape[0]):
            m = min(distances_to_centroids[i])
            for j in range(len(distances_to_centroids[i])):
                if distances_to_centroids[i][j] == m:
                    assignments[i] = int(j)
                    break
        #END
        return assignments

    def compute_centroids(self, X, assignments):
        """
        Given the assignments array for the samples, compute the new centroids
        :param X: data matrix of shape (num_samples, num_features)
        :param assignments: array of shape (num_samples,) where assignment[i] is the current cluster of sample i
        :return: The new centroids array of shape (k, num_features)
        """
        # Workspace 1.4
        centroids = np.zeros((self.k, X.shape[1]))
        #BEGIN 
        for k in range(self.k):
            c = 0
            for i in range(X.shape[0]):
                if assignments[i] == k:
                    for j in range(X.shape[1]):
                        centroids[k][j] += X[i][j]
                    c += 1
            centroids[k] /= c
        #END
        return centroids

    def compute_objective(self, X, assignments):
        return np.sum(np.linalg.norm(X - self.centroids[assignments], axis=1) ** 2)

    def fit(self, X):
        """
        Implement the K-means algorithm here as described above. Loop until the improvement ratio of the objective
        is lower than rtol. At the end of each iteration, save the k-means objective and return the objective values
        at the end

        @param X:
        @return:
        """
        self.centroids = self.initialize_centroids(X)
        objective = np.inf
        assignments = np.zeros((X.shape[0],))
        history = []

        # Workspace 1.5

        while True:
            self.save_snapshot(X, assignments)
            #BEGIN 
            history.append(objective)
            distances = self.compute_distances(X)
            assignments = self.compute_assignments(distances)
            self.centroids = self.compute_centroids(X, assignments)
            objective = self.compute_objective(X, assignments)
            improvement = abs(objective - history[-1]) / history[-1]
            if improvement < self.rtol:
                break
            #END
        return history

    def predict(self, X):
        # Workspace 1.6
        assignments = np.zeros((X.shape[0],))
        #BEGIN 
        distances = self.compute_distances(X)
        assignments = self.compute_assignemnts(distances)
        #END
        return assignments

    def save_snapshot(self, X, assignments):
        """
        Saves plot image of the current asssignments
        """
        if X.shape[1] == 2:
            self.snapshots.append(tests.create_buffer(X, assignments))


def LinearKernel(x1,x2):
    """
    Compute the kernel matrix
    @param x1: array of shape (m1,p)
    @param x2: array of shape(m2,p)        
    @return: K of shape (m1,m2) where K[i,j] = <x1[i], x2[j]>
    """
    # Workspace 2.1
    K = np.zeros((x1.shape[0], x2.shape[0]))
    #BEGIN 
    K = np.dot(x1, np.transpose(x2))
    #END
    return K


def RadialKernel(gamma):
    
    def RadialK(x1, x2):
        """
        Compute the kernel matrix. Hint: computing the squared distances is similar to compute_distances in K-means
        @param x1: array of shape (m1,p)
        @param x2: array of shape(m2,p)
        @return: K of shape (m1,m2) where K[i,j] = K_rad(x1[i],x2[j]) = exp(-gamma * ||x1[i] - x2[j]||^2)
         """
        # Workspace 2.2
        K = np.zeros((x1.shape[0], x2.shape[0]))
        #BEGIN 
        for i in range(x1.shape[0]):
            for j in range(x2.shape[1]):
                K[i][j] = np.exp(-gamma * np.linalg.norm(x1[i] - x2[j])**2)
        #END
        return K
    return RadialK


def PolynomialKernel(c, p):
    
    def PolynomialK(x1, x2):
        """
        Compute the kernel matrix.
        @param x1: array of shape (m1,p)
        @param x2: array of shape(m2,p)
        @return: K of shape (m1,m2) where K[i,j] = (x1[i].x2[j] + c)^p
        """
        # Workspace 2.3
        K = np.zeros((x1.shape[0], x2.shape[0]))
        #BEGIN 
        K = (np.dot(x1,np.transpose(x2))+c)**p
        #END
        return K
    return PolynomialK


fig, axs = plt.subplots(1, 2)
fig.set_figheight(6), fig.set_figwidth(12)
#Workspace 2.5.a
#BEGIN 
svm_radial = SVC(kernel=RadialKernel(2))
for i, dataset in enumerate([binary_blobs, circles]):
    svm_radial.fit(dataset.X_train, dataset.y_train)
    tests.show_decision_surface(svm_radial, dataset.X, dataset.labels, axs[i])
    score = svm_radial.score(dataset.X_test, dataset.y_test)
    axs[i].set_title(score)
plt.show()
#END