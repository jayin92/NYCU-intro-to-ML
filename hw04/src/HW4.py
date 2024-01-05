import numpy as np
from pandas import DataFrame, read_csv
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

degree_ = 3
gamma_ = 0.5

def gram_matrix(X1, X2, kernel_function):
    return np.array([[kernel_function(x1, x2) for x2 in X2] for x1 in X1])

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=degree_):
    return np.power(np.dot(x1, x2) + 1, degree)

def rbf_kernel(x1, x2, gamma=gamma_):
    return np.exp(-gamma * (np.linalg.norm(x1 - x2) ** 2))

if __name__ == "__main__":
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    C_linear = 4
    svc_linear = SVC(kernel='precomputed', C=C_linear)
    svc_linear.fit(gram_matrix(X_train, X_train, linear_kernel), y_train)
    y_pred_linear = svc_linear.predict(gram_matrix(X_test, X_train, linear_kernel))
    print(f"Accuracy of using linear kernel (C = {C_linear}): ", accuracy_score(y_test, y_pred_linear))
    assert accuracy_score(y_test, y_pred_linear) > 0.8

    C_poly = 1
    svc_poly = SVC(kernel='precomputed', C=C_poly)
    svc_poly.fit(gram_matrix(X_train, X_train, polynomial_kernel), y_train)
    y_pred_poly = svc_poly.predict(gram_matrix(X_test, X_train, polynomial_kernel))
    print(f"Accuracy of using polynomial kernel (C = {C_poly}, degree = {degree_}): ", accuracy_score(y_test, y_pred_poly))

    C_rbf = 1
    svc_rbf = SVC(kernel='precomputed', C=C_rbf)
    svc_rbf.fit(gram_matrix(X_train, X_train, rbf_kernel), y_train)
    y_pred_rbf = svc_rbf.predict(gram_matrix(X_test, X_train, rbf_kernel))
    print(f"Accuracy of using rbf kernel (C = {C_rbf}, gamma = {gamma_}): ", accuracy_score(y_test, y_pred_rbf))