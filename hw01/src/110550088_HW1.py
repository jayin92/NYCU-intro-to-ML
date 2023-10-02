# You are not allowed to import any additional packages/libraries.
import numpy as np
import matplotlib.pyplot as plt

from pandas import DataFrame, read_csv


class LinearRegression:
    def __init__(self):
        self.closed_form_weights = None
        self.closed_form_intercept = None
        self.gradient_descent_weights = None
        self.gradient_descent_intercept = None
        self.train_loss = []

    # This function computes the closed-form solution of linear regression.
    def closed_form_fit(self, X, y):
        # Compute closed-form solution.
        # Save the weights and intercept to self.closed_form_weights and self.closed_form_intercept
        X = np.c_[X, np.ones(X.shape[0])]

        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        self.closed_form_weights = beta[:-1]
        self.closed_form_intercept = beta[-1]

    def normalization(self, X):
        # Normalize the data, also return the mean and std of each column.
        # Return the normalized data, mean and std of each column.
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mean) / std
        return X, mean, std

    # This function computes the gradient descent solution of linear regression.
    def gradient_descent_fit(self, X, y, lr, epochs):
        # Compute the solution by gradient descent.
        # Save the weights and intercept to self.gradient_descent_weights and self.gradient_descent_intercept

        # Initialize the weights and intercept.
        beta = np.random.rand(X.shape[1] + 1)
        best = 1e9
        mini_batch_size = 10
        X, mean, std = self.normalization(X)
        X_prime = np.c_[X, np.ones(X.shape[0])]
        # Start training.
        for epoch in range(epochs):
            # Compute the prediction.
            # Mini-batch gradient descent.
            # Randomly sample the data.
            for _ in range(X.shape[0] // mini_batch_size):
                batch = np.random.choice(X_prime.shape[0], mini_batch_size)
                X_batch = X_prime[batch]
                y_batch = y[batch]
                pred = X_batch @ beta
                grad = -(X_batch.T) @ (y_batch - pred) / (X_batch.shape[0])
                beta -= lr * grad

            pred = X_prime @ beta
            eval_loss = self.get_mse_loss(pred, y)

            if best > eval_loss:
                best = eval_loss
                self.gradient_descent_weights = beta[:-1] / std
                self.gradient_descent_intercept = beta[-1] - np.sum(beta[:-1] * (mean / std))

            # if epoch % 10 == 0:
            #     print(f"Epoch {epoch}: {eval_loss}")

            self.train_loss.append(eval_loss)

    # This function compute the MSE loss value between your prediction and ground truth.
    def get_mse_loss(self, prediction, ground_truth):
        loss = np.mean((prediction - ground_truth) ** 2)

        # Return the value.
        return loss

    # This function takes the input data X and predicts the y values according to your closed-form solution.
    def closed_form_predict(self, X):
        # Return the prediction.
        pred = X @ self.closed_form_weights + self.closed_form_intercept
        return pred

    # This function takes the input data X and predicts the y values according to your gradient descent solution.
    def gradient_descent_predict(self, X):
        # Return the prediction.
        pred = X @ self.gradient_descent_weights + self.gradient_descent_intercept
        return pred

    # This function takes the input data X and predicts the y values according to your closed-form solution, 
    # and return the MSE loss between the prediction and the input y values.
    def closed_form_evaluate(self, X, y):
        # This function is finished for you.
        return self.get_mse_loss(self.closed_form_predict(X), y)

    # This function takes the input data X and predicts the y values according to your gradient descent solution, 
    # and return the MSE loss between the prediction and the input y values.
    def gradient_descent_evaluate(self, X, y):
        # This function is finished for you.
        return self.get_mse_loss(self.gradient_descent_predict(X), y)

    # This function use matplotlib to plot and show the learning curve (x-axis: epoch, y-axis: training loss) of your gradient descent solution.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_learning_curve(self):
        loss_plt = plt.plot(self.train_loss)
        # set legend
        plt.legend(loss_plt, ['Training Loss'])
        # set label
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()


# Do not modify the main function architecture.
# You can only modify the arguments of your gradient descent fitting function.
if __name__ == "__main__":
    # Data Preparation
    train_df = DataFrame(read_csv("train.csv"))
    train_x = train_df.drop(["Performance Index"], axis=1)
    train_y = train_df["Performance Index"]
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()

    # Model Training and Evaluation
    LR = LinearRegression()

    LR.closed_form_fit(train_x, train_y)
    print("Closed-form Solution")
    print(f"Weights: {LR.closed_form_weights}, Intercept: {LR.closed_form_intercept}")

    LR.gradient_descent_fit(train_x, train_y, lr=2e-4, epochs=50)
    print("Gradient Descent Solution")
    print(f"Weights: {LR.gradient_descent_weights}, Intercept: {LR.gradient_descent_intercept}")

    test_df = DataFrame(read_csv("test.csv"))
    test_x = test_df.drop(["Performance Index"], axis=1)
    test_y = test_df["Performance Index"]
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    closed_form_loss = LR.closed_form_evaluate(test_x, test_y)
    gradient_descent_loss = LR.gradient_descent_evaluate(test_x, test_y)
    print(f"Error Rate: {((gradient_descent_loss - closed_form_loss) / closed_form_loss * 100):.1f}%")
    LR.plot_learning_curve()
