import numpy as np
import qpsolvers as qps
from SVM import SVM
import itertools
import matplotlib.pyplot as plt
from scipy import sparse

class SoftMarginSVM(SVM):
    """
    Soft Margin Support Vector Machine (SVM) implementation.

    Parameters:
    kernel (string): The kernel function to use. Default is 'rbf'.
    C (float): regularization parameter. Default is 4.
    gamma (float): The gamma scalar for the RBF kernel. Default is 1.
    polly (int): The poly degree for the poly kernel
    max_iter (int): Maximum number of iterations for the QP solver. Default is 2000.
    thresh (float): Threshold for identifying support vectors in the dual form. Default is 0.1.
    """
    def __init__(self, kernel='rbf', c=4, gamma=1, poly=None, max_iter=2000, thresh=0.1):
        primal = False
        self._kernel = kernel
        self._C = c
        self._poly = poly
        self._gamma = gamma
        self._X_support_vectors = None
        self._y_support_vectors = None
        self._support_vector_indices = None
        super().__init__(primal, max_iter, thresh)

    def get_kernel_method(self):
        """
        Retrieves the appropriate kernel function based on the specified kernel type.
        """
        if self._kernel == "poly":
            ker = self.poly_kernel
        elif self._kernel == 'linear':
            ker = self.linear_kernel
        else:
            ker = self.RBF_kernel
        return ker

    def svm_dual_kernel(self, X, y):
        """
        Solves the dual form of the SVM optimization problem.
        """
        ker = self.get_kernel_method()
        N = X.shape[0]
        P = np.empty((N, N))
        for i, j in itertools.product(range(N), range(N)):
            P[i, j] = y[i] * y[j] * ker(X[i, :], X[j, :])
        P = 0.5 * (P + P.T)
        P = sparse.csc_matrix(P)
        q = -np.ones(N)

        G = np.vstack((-np.eye(N), np.eye(N)))
        G = sparse.csc_matrix(G)
        h = np.hstack((np.zeros(N), self._C * np.ones(N)))

        alpha = qps.solve_qp(P, q, G, h, solver='osqp', max_iter=self._max_iter)

        self._alpha = alpha[alpha > self._thresh]
        self._support_vector_indices = np.where(alpha > self._thresh)[0]
        self._X_support_vectors = X[self._support_vector_indices]
        self._y_support_vectors = y[self._support_vector_indices]

    def compute_kernel_matrix(self, X):
        """
        Computes the kernel matrix for the given input samples X.
        """
        ker = self.get_kernel_method()
        num_of_samples = X.shape[0]
        K = np.zeros((num_of_samples, num_of_samples))
        for i in range(num_of_samples):
            for j in range(num_of_samples):
                K[i, j] = ker(X[i, :], X[j, :])
        return K

    def fit(self, X, y):
        """
        Fits the SVM model to the provided data.
        """
        self.svm_dual_kernel(X, y)

    def linear_kernel(self, X, Y):
        """
        Computes the linear kernel between two samples X and y.
        """
        return np.dot(X, Y.T)

    def poly_kernel(self, X, Y):
        """
        Computes the polynomial kernel between two samples X and y.
        """
        return (1 + np.dot(X, Y.T))**self._poly

    def RBF_kernel(self, X, Y):
        """
        Computes the RBF kernel between two samples X and y.
        """
        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        if len(Y.shape) == 1:
            Y = Y[np.newaxis, :]

        norm_squared = np.sum((X[:, np.newaxis] - Y[np.newaxis, :])**2, axis=2)
        return np.exp(-self._gamma * norm_squared)

    def plot_title(self):
        """
        Generates and sets the plot title based on the kernel type and its parameters.
        """
        if self._kernel == "poly":
            txt = "SVM Decision Boundary with Polynomial Kernel (Degree = {degree})"
            plt.title(txt.format(degree=self._poly))
        elif self._kernel == 'linear':
            txt = "SVM Decision Boundary with Linear Kernel"
            plt.title(txt)
        else:
            txt = "SVM Decision Boundary with RBF Kernel (Gamma = {gamma})"
            plt.title(txt.format(gamma=self._gamma))

    def plot_classifier_z_kernel(self, X, y, s=None):
        """
        Plots the decision boundary and support vectors of the SVM model in the feature space.
        """
        self.plot_title()
        ker = self.get_kernel_method()

        # Compute range for plotting
        x_min = np.amin(X[:, 0])
        x_max = np.amax(X[:, 0])
        y_min = np.amin(X[:, 1])
        y_max = np.amax(X[:, 1])

        xx = np.linspace(x_min, x_max)
        yy = np.linspace(y_min, y_max)

        xx, yy = np.meshgrid(xx, yy)

        N = self._X_support_vectors.shape[0]
        z = np.zeros(xx.shape)
        for i, j in itertools.product(range(xx.shape[0]), range(xx.shape[1])):
            z[i, j] = sum([self._y_support_vectors[k] * self._alpha[k]
                           * ker(self._X_support_vectors[k, :], np.array([xx[i, j], yy[i, j]])) for k in range(N)])

        plt.rcParams["figure.figsize"] = [15, 10]

        # Plot decision boundary
        plt.contour(xx, yy, z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'], linestyles=['--', '-', '--'])
        self.highlight_support_vectors(X)
        self.plot_data(X, y, s=s)
        plt.show()

    def predict(self, X):
        """
        Predicts the labels for input data points.
        """
        decision = self.decision_function(X)
        return np.sign(decision)

    def decision_function(self, X):
        """
        Computes the decision function values for input data points.
        """
        ker = self.get_kernel_method()
        K = ker(self._X_support_vectors, X)
        return np.dot(self.alpha * self._y_support_vectors, K)

    def score(self, X, y):
        """
        Computes the accuracy score of the SVM model.
        """
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
