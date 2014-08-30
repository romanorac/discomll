"""
Locally Weighted Linear Regression
"""

import numpy as np

class Locally_Weighted_Linear_Regression():
    """Iterative version of Locally Weighted Linear Regression """

    def fit(self, X, y, X_est, tau = 1):
        """
        Fit X_est with Locally weighted linear regression according to X and y

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        X_est : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Estimation vectors, where n_samples is the number of samples and
            n_features is the number of features.

        tau : float, tau >= 0
            The the bandwidth parameter tau controls how quickly the weight of a training example falls off with distance of its x(i) from the query point x.

        References
        ----------
        CS229 Lecture notes1, Chapter 3 Locally weighted linear regression, Prof. Andrew Ng
        http://cs229.stanford.edu/notes/cs229-notes1.pdf
        """

        thetas = [] #init estimated thetas
        estimation = [] #init estimation of X_est

        if tau <= 0:
            print "tau should be greater than 0."
            return [],[]

        for x in X_est: #for every sample in X_est
            #calculate weights that depend on the particular vector x 
            weights =  np.exp((-(X - x)*(X - x)).sum(axis = 1)/(2 * tau**2))            
            W = np.diag(weights) #diagonal matrix with weights
            x_W = np.dot(X.T, W) 
            A = np.dot(x_W, X)
            b = np.dot(x_W, y)
            thetas.append(np.linalg.lstsq(A,b)[0])# calculate thetas for given x with: A^-1 * b
            estimation.append(np.dot(x, thetas[-1])) # calculate estimation for given x and thetas 

        return thetas, estimation

if __name__ == '__main__':
    """ 
    Example was taken from: 
    http://www.dsplog.com/2012/02/05/weighted-least-squares-and-locally-weighted-linear-regression/  
    """
    import load_dataset
    import matplotlib.pyplot as plt
    
    d = load_dataset.Data() #Load data
    X, y = d.load_lwlr()
    
    lwlr = Locally_Weighted_Linear_Regression()
    taus = [1, 10, 25]
    plt.scatter(X[:,1], y) #Plot train data
    
    color = ["r","g", "b"]
    for i, tau in enumerate(taus):
        thetas, estimation = lwlr.fit(X, y, X, tau = tau)
        plt.plot(X[:,1] ,estimation, c = color[i]) #Plot prediction

    plt.show()



