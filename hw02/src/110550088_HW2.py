# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class LogisticRegression:
    def __init__(self, learning_rate=0.1, iteration=100):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.intercept = None

    # This function computes the gradient descent solution of logistic regression.
    def fit(self, X, y):
        # Initialize the weights and intercept with zeros.
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mean) / std
        self.weights = np.zeros(X.shape[1])
        self.intercept = 0

        # Update the weights and intercept iteratively.
        for i in range(self.iteration):
            # Compute the gradient of the loss function with respect to the weights and intercept.
            dw, db = self.compute_gradient(X, y)
            # Update the weights and intercept.
            self.weights -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db
        
        tmp_weights = self.weights.copy()
        self.weights = self.weights / std
        self.intercept = self.intercept - (tmp_weights * (mean / std)).sum()

    # This function computes the gradient of the loss function with respect to the weights and intercept.
    def compute_gradient(self, X, y):
        # Compute the probability of being class 1.
        y_hat = X @ self.weights + self.intercept
        y_pred = self.sigmoid(y_hat)

        # This is simplified by the derivative of CE loss function
        db = -(y-y_pred).mean()
        dw = -((X.T) @ (y-y_pred))
        dw /= X.shape[0]

        return dw, db

    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        # Compute the probability of being class 1.
        y_pred = self.sigmoid(X @ self.weights + self.intercept)
        # Round the probability to the nearest integer.
        y_pred = np.round(y_pred)

        return y_pred

    # This function computes the value of the sigmoid fㄥunction.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
       
    def logistic(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):
        # Compute the means of the two classes.
        self.m0 = X[y == 0].mean(axis=0)
        self.m1 = X[y == 1].mean(axis=0)

        # Compute the within-class scatter matrix.
        self.sw = ((X[y == 0] - self.m0).T @ (X[y == 0] - self.m0)) + ((X[y == 1] - self.m1).T @ (X[y == 1] - self.m1))

        # Compute the between-class scatter matrix
        self.sb = np.outer((self.m0 - self.m1), (self.m0 - self.m1))

        # Compute the solution of Fisher's Linear Discriminant.
        self.w = np.linalg.inv(self.sw) @ (self.m1 - self.m0)
        self.w = self.w / np.linalg.norm(self.w)

        # Compute the slope of the projection line.
        self.slope = self.w[1] / self.w[0]


    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def predict(self, X):
        # Compute the projected result of the testing data.
        y_pred = X @ self.w

        # Compute the projected means of the two classes.
        m0 = self.m0 @ self.w
        m1 = self.m1 @ self.w

        # Round the probability to the nearest integer.
        y_pred = np.round(y_pred)

        # Predict the class label.
        y_pred[y_pred < (m0 + m1) / 2] = 0
        y_pred[y_pred >= (m0 + m1) / 2] = 1

        return y_pred

    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_projection(self, X):

        y = self.predict(X)
        plt.figure(figsize=(10, 10))
        plt.axes().set_aspect('equal')
        # set figure size
        plt.xlim(-40, 250)
        plt.ylim(-40, 250)
        # Plot the testing data, positive samples in blue, negative samples in red.
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c="r")
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c="b")

        b = 200

        # Plot the projection line.
        x = np.linspace(-40, 250, 100)
        plt.plot(x, self.slope * x + b, c="blue")

        # Calculate the projected point of each sample and connect the projected point with the original point.
        u = np.array([0, b])
        for i in range(X.shape[0]):
            pos = u + np.dot(X[i] - u, self.w) * self.w
            plt.scatter(pos[0], pos[1], c=("r" if y[i] == 0 else "b"), alpha=0.3)
            # draw line connect two points
            plt.plot([X[i][0], pos[0]], [X[i][1], pos[1]], c="purple", alpha=0.3)

        # Show title
        plt.title(f"Projection Line: w={round(self.slope,2)}, b={round(b,2)}")

        # Show the plot.
        plt.show()

     
# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))

# Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    LR = LogisticRegression(learning_rate=0.01, iteration=1000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Part 1: Logistic Regression")
    print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    print(f"Accuracy: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

# Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"

    # Plot the projection line.
    FLD.plot_projection(X_test)
