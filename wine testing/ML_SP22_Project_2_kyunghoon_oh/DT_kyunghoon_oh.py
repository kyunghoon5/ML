

import pandas as pd
import numpy as np
import random
from collections import Counter
import seaborn as sns
#https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb

from sklearn import datasets, __all__

from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None
    

class DecisionTreeModel:

    def __init__(self, max_depth=100, criterion = 'gini', min_samples_split=2, impurity_stopping_threshold = 1):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.impurity_stopping_threshold = impurity_stopping_threshold
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.Series):

        """
        to check for y if its categorical or numeric, and if its categorical you can change using Ordenial encoding to change it to numeric value and then pass into
        _numpy andthe rest will work itself out
        since the original code is based on numpy, and it passes in a numpy, it will owrk as intended
        """
        self._fit(X.to_numpy(), y.to_numpy())


    def predict(self, X: pd.DataFrame):

        """
        to check for X if its categorical or numeric, and if its categorical you can change using Ordenial encoding to change it to numeric value and then pass into
        _numpy andthe rest will work itself out
        since the original code is based on numpy, and it passes in a numpy, it will work as intended
        """
        return self._predict(X.to_numpy())

    def _fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

    def _is_finished(self, depth):
        # TODO: for graduate students only, add another stopping criteria
        # modify the signature of the method if needed
        if (depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        # end TODO
        return False
    
    def _is_homogenous_enough(self):
        # TODO: for graduate students only
        result = False
        # end TODO
        return result
                              
    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        if self._is_finished(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)

        # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)

    def _gini(self, y):

        """
        gini index for DT
        y will be represented as numerical
        """
        proportions = np.bincount(y) / len(y)
        gini = np.sum([p * (1 - p) for p in proportions if p > 0])
        return gini
    
    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        
    def _create_split(self, X, thresh):
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _information_gain(self, X, y, thresh):
        '''
        to get gini or entorpy in the critertion
        '''

        if self.criterion == 'gini':
            parent_loss = self._gini(y)
        else:
            parent_loss = self._entropy(y)

        left_idx, right_idx = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0:
            return 0

        child_loss = (n_left / n) * self._gini(y[left_idx]) + (n_right / n) * self._gini(y[right_idx])

        return parent_loss - child_loss

    def _best_split(self, X, y, features):
        '''TODO: add comments here
        looping each value and calculate score. if score is higher then split score it return feat and threshold
        '''
        split = {'score':- 1, 'feat': None, 'thresh': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']
    
    def _traverse_tree(self, x, node):
        '''TODO: add some comments here
        traverse from root then check each values. if value is equal or less than threshold it travers the left node. e;se right
        '''
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestModel(object):


    def __init__(self, n_estimators: int, max_depth=100, criterion='gini', min_samples_split=2,
                     impurity_stopping_threshold=1):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.criterion = criterion
            self.min_samples_split = min_samples_split
            self.impurity_stopping_threshold = impurity_stopping_threshold
            self.forest = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for i in range(0, self.n_estimators):
            # shuffles the data
            idxs = np.random.choice(len(y), replace=True, size=len(y))
            # Creates a tree
            tree = DecisionTreeModel(max_depth=self.max_depth, criterion=self.criterion,
                                     min_samples_split=self.min_samples_split,
                                     impurity_stopping_threshold=self.impurity_stopping_threshold)
            # Fit the tree while passing in the shuffled indexes
            tree.fit(X.iloc[idxs], y.iloc[idxs])
            # Append the tree to the array of `forests`
            self.forest.append(tree)

    def _common_result(self, values: list):
        return np.array([Counter(col).most_common(1)[0][0] for col in zip(*values)])

    def predict(self, X: pd.DataFrame):
        tree_values = []
        for tree in self.forest:
            tree_values.append(tree.predict(X))
        return self._common_result(tree_values)

    

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def _count_occurence(y_pred):

    one, zero = 0, 0

    for i in y_pred:
        if i == 1:
            one += 1
        else:
            zero += 1

    return zero, one


def classification_report(y_test, y_pred):
    # assumption data will be a 1d array of same length i.e y_pred and y_test
    # calculate precision, recall, f1-score
    top, bottom = confusion_matrix(y_test, y_pred)
    tp, fn = top
    fp, tn = bottom

    precision0 = (tn / (tn + fn))
    precision1 = (tp / (tp + fp))

    recall0 = (tn / (tn + fp))
    recall1 = (tp / (tp + fn))

    f1_score0 = 2 * (recall0 * precision0) / (recall0 + precision0)
    f1_score1 = 2 * (recall1 * precision1) / (recall1 + precision1)

    support0, support1 = _count_occurence(y_pred)

    result = f'''                    precision    recall    f1-score    support
        0               {"%.2f" % round(precision0, 2)}        {"%.2f" % round(recall0, 2)}        {"%.2f" % round(f1_score0, 2)}        {support0}
        1               {"%.2f" % round(precision1, 2)}        {"%.2f" % round(recall1, 2)}        {"%.2f" % round(f1_score1, 2)}        {support1}
        accuracy                                {"%.2f" % round(accuracy_score(y_test, y_pred), 2)}        {support0 + support1}
        macro avg       {"%.2f" % round((precision0 + precision1) / 2, 2)}        {"%.2f" % round((recall0 + recall1) / 2, 2)}        {"%.2f" % round((f1_score0 + f1_score1) / 2, 2)}        {support0 + support1}
        weighted avg    {"%.2f" % round(((precision0 * support0) + (precision1 * support1)) / (support0 + support1), 2)}        {"%.2f" % round(((recall0 * support0) + (recall1 * support1)) / (support0 + support1), 2)}        {"%.2f" % round(((f1_score0 * support0) + (f1_score1 * support1)) / (support0 + support1), 2)}        {support0 + support1}
        '''
    return result





def confusion_matrix(y_test, y_pred):
    # return the 2x2 matrix
        # return the 2x2 matrix
    tp, fn, fp, tn = 0, 0, 0, 0
    for i, val in y_test.reset_index(drop=True).iteritems():
        p_val = y_pred[i]
        if val == 1 and p_val == 1:
            tp += 1
        elif val == 1 and p_val == 0:
            fn += 1
        elif val == 0 and p_val == 1:
            fp += 1
        else:
            tn += 1
    result = np.array([[tp, fn], [fp, tn]])
    return (result)


def _test():
    
    df = pd.read_csv('breast_cancer.csv')
    
    #X = df.drop(['diagnosis'], axis=1).to_numpy()
    #y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1).to_numpy()

    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    clf = DecisionTreeModel(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)


if __name__ == "__main__":
    _test()
