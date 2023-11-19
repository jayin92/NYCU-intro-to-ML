# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# This function computes the gini impurity of a label array.
def gini(y):
    pass

# This function computes the entropy of a label array.
def entropy(y):
    pass
        
# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth 
    
    # This function computes the impurity based on the criterion.
    def impurity(self, y):
        if self.criterion == 'gini':
            return gini(y)
        elif self.criterion == 'entropy':
            return entropy(y)
    
    # This function fits the given data using the decision tree algorithm.
    def fit(self, X, y):
        pass
    
    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        pass
    
    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, columns):
        pass

# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        self.criterion = criterion 
        self.n_estimators = n_estimators

    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        pass

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        pass

# Do not modify the main function architecture.
# You can only modify the value of the random seed and the the arguments of your Adaboost class.
if __name__ == "__main__":
# Data Loading
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

# Set random seed to make sure you get the same result every time.
# You can change the random seed if you want to.
    np.random.seed(0)

# Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))

# AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    ada = AdaBoost(criterion='gini', n_estimators=200)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


    
