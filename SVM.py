import matplotlib
import numpy as np
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, primal, max_iter=2000, thresh=0.1):
        self._max_iter = max_iter
        self._primal = primal
        self._thresh = thresh
        self._weights = None
        self._alpha = None
        self._X_support_vectors = None
        self._support_vector_indices = None

    @property
    def weights(self):
        """
        Returns the weights of the SVM model.
        """
        return self._weights

    @property
    def alpha(self):
        """
        Returns the dual variables (alpha) of the SVM model if in dual mode.
        """
        return self._alpha

    @property
    def support_vectors(self):
        """
        Returns the support vectors of the SVM model if in dual mode.
        """
        if self._primal:
            return None
        return self._X_support_vectors

    def plot_data(self, X, y, s=None):
        """
        Plots the data points with different colors for different labels.
        """
        plt.scatter(X[:, 0], X[:, 1], c=y, s=s, cmap=matplotlib.colors.ListedColormap(['red', 'blue']))

    def highlight_support_vectors(self, X):
        """
        Highlights the support vectors on the plot.
        """
        if self._support_vector_indices is not None:
            plt.scatter(X[self._support_vector_indices, 0], X[self._support_vector_indices, 1], s=300, linewidth=3,
                        facecolors='none',
                        edgecolors='k')

    def score(self, X, y):
        """
        Computes the accuracy score of the SVM model.
        """
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

    def predict(self, X):
        """
        Predicts the labels for input data points.
        """
        return np.sign(np.dot(X, self._weights))
