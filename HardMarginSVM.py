import matplotlib
from SVM import SVM
import numpy as np
import qpsolvers as qps
from scipy import sparse
import matplotlib.pyplot as plt


class HardMarginSVM(SVM):
    """
    Hard Margin Support Vector Machine (SVM) implementation.

    Parameters:
    primal (bool): If True, uses the primal form of SVM. Otherwise, uses the dual form.
    max_iter (int): Maximum number of iterations for the QP solver. Default is 2000.
    thresh (float): Threshold for identifying support vectors in the dual form. Default is 0.1.
    """
    def __init__(self, primal, max_iter=2000, thresh=0.1):
        super().__init__(primal, max_iter, thresh)

    def svm_primal(self, X, y):
        """
        Solves the primal form of the SVM optimization problem.
        """
        N = X.shape[0]
        n = X.shape[1]
        P = np.eye(n)
        P = sparse.csc_matrix(P)
        q = np.zeros(n)
        G = -np.diag(y) @ X
        G = sparse.csc_matrix(G)
        h = -np.ones(N)
        self._weights = np.array(qps.solve_qp(P, q, G, h, solver='osqp', max_iter=self._max_iter))

    def svm_dual(self, X, y):
        """
        Solves the dual form of the SVM optimization problem.
        """
        N = X.shape[0]
        G = np.diag(y) @ X
        P = 0.5 * G @ G.T
        P = sparse.csc_matrix(P)
        q = -np.ones(N)
        GG = -np.eye(N)
        GG = sparse.csc_matrix(GG)
        h = np.zeros(N)
        alpha = np.array(qps.solve_qp(P, q, GG, h, solver='osqp', max_iter=self._max_iter))
        self._weights = G.T @ alpha
        self._alpha = alpha
        self._X_support_vectors = np.argwhere(np.abs(self._alpha) > self._thresh).reshape(-1)
        self._support_vector_indices = np.where(alpha > self._thresh)[0]


    def fit(self, X, y):
        """
        Fits the SVM model to the provided data.
        """
        if self._primal:
            self.svm_primal(X, y)
        else:
            self.svm_dual(X, y)

    def plot_data_with_decision_boundary(self, X, y):
        """
        Plots the data points, decision boundary, and margins.
        """
        self.plot_data(X, y)

        # Generate points for the decision boundary line
        lx = np.linspace(np.amin(X[:, 0]), np.amax(X[:, 0]), 60)

        # Decision boundary
        ly = [(-self._weights[-1] - self._weights[0] * p) / self._weights[1] for p in lx]
        plt.plot(lx, ly, color='black')

        # Margins
        ly1 = [(-self._weights[-1] - self._weights[0] * p - 1) / self._weights[1] for p in lx]
        plt.plot(lx, ly1, "--", color='red')

        ly2 = [(-self._weights[-1] - self._weights[0] * p + 1) / self._weights[1] for p in lx]
        plt.plot(lx, ly2, "--", color='blue')

        # Highlight support vectors if using the dual form
        if not self._primal:
            self.highlight_support_vectors(X)

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Data Points and SVM Decision Boundary')
        plt.show()
