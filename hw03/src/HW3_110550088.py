# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

np.random.seed(156)

# This function computes the gini impurity of a label array.
def gini(y):
    if len(y) == 0:
        return 0

    prob = np.bincount(y) / len(y)
    np.seterr(divide = 'ignore') 
    res = 1 - np.sum(np.square(prob))
    np.seterr(divide = 'warn')
    return res

# This function computes the entropy of a label array.
def entropy(y):
    # if len(y) == 0:
        # return 0
    prob = np.bincount(y) / len(y) + 1e-10

    # np.seterr(invalid='ignore', divide='ignore')
    res = -np.sum(prob * np.log2(prob))
    # np.seterr(invalid='warn', divide='warn')

    return res

        
class Node:
    def __init__(self, feature, threshold, left, right, impurity, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.impurity = impurity
        self.value = value
    
    def is_leaf(self):
        return self.value is not None

    def predict(self, x):
        if self.is_leaf():
            return self.value
        else:
            if x[self.feature] <= self.threshold:
                return self.left.predict(x)
            else:
                return self.right.predict(x)
    


# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth 
    
    # This function computes the impurity based on the criterion.
    def impurity(self, y):
        # if len(y) == 0:
            # return 0
        if self.criterion == 'gini':
            return gini(y)
        elif self.criterion == 'entropy':
            return entropy(y)
    
    # This function fits the given data using the decision tree algorithm.
    def build(self, X, y, depth):
        if depth == self.max_depth or self.impurity(y) == 0:
            if len(y) == 0:
                return None
            return Node(None, None, None, None, self.impurity(y), np.argmax(np.bincount(y)))
        else:
            min_impurity = np.inf
            min_feature = None
            min_threshold = None
            min_left = None
            min_right = None
            for feature in range(X.shape[1]):
                for threshold in np.unique(X[:, feature]):
                    left = y[X[:, feature] <= threshold] 
                    right = y[X[:, feature] > threshold]
                    impurity = self.impurity(left) * len(left) / len(y) + self.impurity(right) * len(right) / len(y)

                    if impurity < min_impurity:
                        min_impurity = impurity
                        min_feature = feature
                        min_threshold = threshold
                        min_left = left
                        min_right = right
            return Node(
                feature=min_feature,
                threshold=min_threshold,
                left=self.build(X[X[:, min_feature] <= min_threshold], min_left, depth + 1),
                right=self.build(X[X[:, min_feature] > min_threshold], min_right, depth + 1),
                impurity=min_impurity
            )
    
    def random_build(self, X, y, depth):
        if depth == self.max_depth or self.impurity(y) == 0:
            if len(y) == 0:
                return None
            return Node(None, None, None, None, self.impurity(y), np.argmax(np.bincount(y)))
        else:
            feature = np.random.choice(X.shape[1])
            threshold = np.random.choice(np.unique(X[:, feature]))
            left = y[X[:, feature] <= threshold]
            right = y[X[:, feature] > threshold]
            impurity = self.impurity(left) * len(left) / len(y) + self.impurity(right) * len(right) / len(y)

            return Node(
                feature=feature,
                threshold=threshold,
                left=self.random_build(X[X[:, feature] <= threshold], left, depth + 1),
                right=self.random_build(X[X[:, feature] > threshold], right, depth + 1),
                impurity=impurity
            )
            min_impurity = np.inf
            min_feature = None
            min_threshold = None
            min_left = None
            min_right = None
            feature = np.random.choice(X.shape[1])
            for threshold in np.unique(X[:, feature]):
                left = y[X[:, feature] <= threshold]
                right = y[X[:, feature] > threshold]
                impurity = self.impurity(left) * len(left) / len(y) + self.impurity(right) * len(right) / len(y)

                if impurity < min_impurity:
                    min_impurity = impurity
                    min_feature = feature
                    min_threshold = threshold
                    min_left = left
                    min_right = right
            return Node(
                feature=min_feature,
                threshold=min_threshold,
                left=self.build(X[X[:, min_feature] <= min_threshold], min_left, depth + 1),
                right=self.build(X[X[:, min_feature] > min_threshold], min_right, depth + 1),
                impurity=min_impurity
            )

    def fit(self, X, y):
        self.root = self.build(X, y, 0)

    def random_fit(self, X, y):
        self.root = self.random_build(X, y, 0)

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        return np.array([self.root.predict(x) for x in X])
    
    def ada_predict(self, X):
        return np.array([1 if self.root.predict(x) == 1 else -1 for x in X])
    
    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, columns):
        pass

# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        self.criterion = criterion 
        self.n_estimators = n_estimators
        self.d = []
        self.trees = []
        self.y_preds = []
        self.alpha = []
        self.clfs = []

    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        self.d = np.ones(len(X)) / len(X)
        y_prime = np.array([1 if y[i] == 1 else -1 for i in range(len(y))])

        for t in range(self.n_estimators):
            min_error = np.inf
            best_clf = None
            best_y_pred = None
            for _ in range(10):
                clf = DecisionTree(criterion=self.criterion, max_depth=1)
                clf.random_fit(X, y)
                y_pred = clf.predict(X)
                # print(f"{(y_pred == y).sum()}/{len(y)}")
                error = np.sum(self.d * (y_pred != y))

                if error < min_error:
                    min_error = error
                    best_clf = clf
                    best_y_pred = y_pred

            # print the feature and threshold of the best clf
            # print(f"min_error: {min_error}, clf_root: {best_clf.tree_.feature[0]}, clf_threshold: {best_clf.tree_.threshold[0]}")

            # print(f"{t}: min_error: {min_error}, clf_root: {best_clf.root.feature}, clf_threshold: {best_clf.root.threshold}")
            # make y_pred = 0 to be -1
            best_y_pred = np.array([1 if best_y_pred[i] == 1 else -1 for i in range(len(best_y_pred))])
            # print(y * best_y_pred)
            self.clfs.append(best_clf)
            self.alpha.append(0.5 * np.log((1 - min_error) / min_error))
            new_d = self.d * np.exp(-self.alpha[t] * y_prime * best_y_pred)
            new_d = new_d / new_d.sum()
            self.d = new_d
            # print(self.d[t + 1])
            # print(self.d[t + 1].max())

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        # print([self.alpha[t] * self.clfs[t].ada_predict(X) for t in range(self.n_estimators)])
        # for i, clf in enumerate(self.clfs):
            # print(f"{i}: clf_root: {clf.root.feature}, clf_threshold: {clf.root.threshold}")
        # print(np.sum([self.alpha[t] * self.clfs[t].ada_predict(X) for t in range(self.n_estimators)], axis=0))
        return np.sum([self.alpha[t] * self.clfs[t].ada_predict(X) for t in range(self.n_estimators)], axis=0) > 0

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
    ada = AdaBoost(criterion='gini', n_estimators=50)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    # print(y_pred)
    print("Accuracy:", accuracy_score(y_test, y_pred))


    
