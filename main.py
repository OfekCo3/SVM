from HardMarginSVM import HardMarginSVM
from SoftMarginSVM import SoftMarginSVM
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def csv_to_np(str_file_name):
    df = pd.read_csv(str_file_name)
    X = np.array(df.iloc[:, :-1].values)
    y = np.array(df.iloc[:, -1].values)
    unique_labels = np.unique(y)
    if set(unique_labels).issubset({0, 1}):
        y = y * 2 - 1  # transpose [0,1] labels to [-1,1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


def evaluate_kernels(X, y, kernels, plot_result=False):
    results = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for kernel, params in kernels.items():
        errors = []

        for param in params:
            if kernel == 'poly':
                svm = SoftMarginSVM(c=param['c'], kernel='poly', poly=param['degree'], thresh=param['thresh'])
                label = f"Poly Degree {param['degree']}"
            elif kernel == 'rbf':
                svm = SoftMarginSVM(c=param['c'], kernel='rbf', gamma=param['gamma'], thresh=param['thresh'])
                label = f"RBF Gamma {param['gamma']}"
            elif kernel == 'linear':
                svm = SoftMarginSVM(c=param['c'], kernel='linear', thresh=param['thresh'])
                label = "Linear"

            svm.fit(X_train, y_train)
            score = svm.score(X_test, y_test)

            if plot_result:
                svm.plot_classifier_z_kernel(X, y)

            error = 1 - score
            errors.append(error)
            print(f"{label} Error: {error}")

        results[kernel] = errors

    return results


def plot_errors(errors):
    plt.figure(figsize=(10, 5))

    for kernel, kernel_errors in errors.items():
        if kernel == 'poly':
            plt.plot(range(2, 2 + len(kernel_errors)), kernel_errors, marker='o', label='Polynomial Kernel')
            for i, error in enumerate(kernel_errors, start=2):
                plt.text(i, error, f"{error:.2f}", ha='center', va='bottom')
        elif kernel == 'rbf':
            gamma_values = np.arange(0.5, 0.5 + 0.5 * len(kernel_errors), 0.5)
            plt.plot(gamma_values, kernel_errors, marker='o', label='RBF Kernel')
            for i, error in enumerate(kernel_errors):
                gamma_value = 0.5 * (i + 1)
                plt.text(gamma_value, error, f"{error:.2f}", ha='center', va='bottom')
        elif kernel == 'linear':
            plt.scatter([1], kernel_errors, color='green', label='Linear Kernel', zorder=5)
            plt.text(1, kernel_errors[0], f"{kernel_errors[0]:.2f}", ha='center', va='bottom', color='green')

    plt.xlabel('Degree / Gamma')
    plt.ylabel('Error Rate')
    plt.title('Error Rate Comparison for Polynomial, RBF, and Linear Kernels')
    plt.xticks(
        list(range(2, 2 + len(errors['poly']))) + list(np.arange(0.5, 0.5 + 0.5 * len(errors['rbf']), 0.5)) + [1])
    plt.legend()
    plt.grid(True)
    plt.show()


def first_question():
    X, y = csv_to_np("simple_classification.csv")
    X = np.c_[X, np.ones(X.shape[0])]  # add constant
    print("------------- first question -------------")
    print("Hard SVM - primal model")
    primal_svm = HardMarginSVM(primal=True)
    primal_svm.fit(X, y)
    print("The Weights: ", primal_svm.weights)
    primal_svm.plot_data_with_decision_boundary(X, y)

    print("Hard SVM - dual model")
    dual_svm = HardMarginSVM(primal=False)
    dual_svm.fit(X, y)
    print("The Weights: ", dual_svm.weights)
    print("The Support vectors: ", dual_svm.support_vectors)
    dual_svm.plot_data_with_decision_boundary(X, y)


def second_question():
    print("--------- Second and Third questions ---------")

    X, y = csv_to_np("simple_nonlin_classification.csv")

    kernels = {
        'poly': [{'c': 10, 'degree': i, 'thresh': 0.1} for i in range(2, 5)],
        'linear': [{'c': 1, 'thresh': 0.1}],
        'rbf': [{'c': 7, 'gamma': i, 'thresh': 0.5} for i in np.arange(0.5, 2.5, 0.5)]
    }

    errors = evaluate_kernels(X, y, kernels, plot_result=True)
    plot_errors(errors)


def fourth_question():
    print("------------- Fourth question -------------")

    X, y = csv_to_np("Processed Wisconsin Diagnostic Breast Cancer.csv")

    kernels = {
        'linear': [{'c': 10, 'thresh': 0.01}],
        'poly': [{'c': 1, 'degree': i, 'thresh': 0.001} for i in range(2, 5)],
        'rbf': [{'c': 7, 'gamma': i, 'thresh': 0.5} for i in np.arange(0.5, 2, 0.5)]
    }

    errors = evaluate_kernels(X, y, kernels)
    plot_errors(errors)


if __name__ == '__main__':
    first_question()
    second_question()
    fourth_question()
