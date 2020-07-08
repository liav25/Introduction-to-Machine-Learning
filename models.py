import numpy as np
from abc import ABC, abstractmethod


class Classifier(ABC):

    def score(self, X, y):
        """
        :param X:
        :param y:
        :return:
        """
        keys = ["num_samples", "error", "accuracy", "FPR",
                "TPR", "precision", "recall"]

        pred = self.predict(X)

        P = (pred > 0).sum()
        N = (pred <= 0).sum()

        TN = ((pred < 0) & (y < 0)).sum()
        FP = ((pred > 0) & (y < 0)).sum()

        TP = ((pred > 0) & (y > 0)).sum()
        FN = ((pred < 0) & (y > 0)).sum()

        num_samp = len(y)
        err = (FP + FN) / (P + N)
        acc = (TP + TN) / (P + N)
        precision = TP / (TP + FP)
        tpr = TP / P
        recall = tpr
        fpr = FP / N

        values = [num_samp, err, acc, fpr, tpr, precision, recall]

        score = dict(zip(keys, values))

        return score

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass


class Perceptron(Classifier):

    def __init__(self):
        self.model = None
        self.name = "Perceptron"

    def fit(self, X, y):
        """
        fit the perceptron w vector.
        :param X:
        :param y:
        """
        # number of parameters, observations
        p, m = X.shape[0] + 1, X.shape[1]

        # initialize w vector
        self.model = np.zeros(p)

        X = np.row_stack([np.ones(m), X])

        # while exist i s.t (y_i*<w, x_i>) <=0
        while True:
            z = (self.model @ X)
            scores = y * z
            if (scores <= 0).any():
                # get the first i s.t. (y_i*<w, x_i>) <=0
                i = np.argmax(scores <= 0)
                # update perceptron
                self.model += y[i] * X[:, i]
            else:
                break

    def predict(self, X):
        # number of parameters, observations
        p, m = X.shape[0] + 1, X.shape[1]
        X = np.row_stack([np.ones(m), X])

        return np.sign(self.model @ X)


class LDA(Classifier):

    def __init__(self):
        self.name = "LDA"
        self.NEG, self.POS = -1, 1
        self.pr_pos, self.pr_neg = None, None
        self.mu_pos, self.mu_neg = None, None
        self.cov_inv = None

    def fit(self, X, y):
        pr_y = (y > 0).mean()
        self.pr_pos, self.pr_neg = pr_y, 1 - pr_y
        self.mu_pos = X[:, y > 0].mean(axis=1)
        self.mu_neg = X[:, y <= 0].mean(axis=1)
        self.cov_inv = np.linalg.pinv(np.cov(X))

    def predict(self, X):
        mu = self.mu_pos
        d1 = (X.T @ self.cov_inv @ mu) - 0.5 * (mu.T @ self.cov_inv @ mu) + np.log(self.pr_pos)
        mu = self.mu_neg
        d2 = (X.T @ self.cov_inv @ mu) - 0.5 * (mu.T @ self.cov_inv @ mu) + np.log(self.pr_neg)
        y = []
        for i in range(len(d1)):
            if d1[i] > d2[i]:
                y.append(self.POS)
            else:
                y.append(self.NEG)
        return np.array(y)

    # def predict(self, X):
    #     output_lst = [1, -1]
    #     y_hat = np.apply_along_axis(self.predict_single, 0, X, output_lst)
    #     print(y_hat, y_hat.shape)
    #     return y_hat

    # def predict_single(self, x, output_lst):
    #     mu_lst = [self.mu_pos, self.mu_neg]
    #     score_lst = []
    #     for i in range(len(mu_lst)):
    #         mu = mu_lst[i]
    #         d = (x.T @ self.cov_inv @ mu) - 0.5 * (mu.T @ self.cov_inv @ mu) + np.log(self.pr_y_lst[i])
    #         score_lst.append(d)
    #
    #     max_i = np.argmax(score_lst)
    #     return output_lst[max_i]


class SVM(Classifier):

    def __init__(self, c=1e10):
        from sklearn.svm import SVC
        self.name = "SVM"
        self.svm = SVC(C=c, kernel='linear')
        self.model = None

    def fit(self, X, y):
        self.svm.fit(X.T, y)
        self.model = np.append(self.svm.intercept_, self.svm.coef_[0])

    def predict(self, X):
        y_hat = self.svm.predict(X.T)
        return y_hat


class Logistic(Classifier):

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.name = "Logistic"
        self.logistic = LogisticRegression(solver='liblinear')
        self.model = None

    def fit(self, X, y):
        self.logistic.fit(X.T, y)
        self.model = np.append(self.logistic.intercept_, self.logistic.coef_[0])

    def predict(self, X):
        y_hat = self.logistic.predict(X.T)
        return y_hat


class DecisionTree(Classifier):
    def __init__(self, maxdepth=100):
        from sklearn.tree import DecisionTreeClassifier
        self.name = "Decision Tree"
        self.tree = DecisionTreeClassifier(max_depth=maxdepth)

    def fit(self, X, y):
        self.tree.fit(X.T, y)

    def predict(self, X):
        y_hat = self.tree.predict(X.T)
        return y_hat


class KNN(Classifier):
    def __init__(self, neighbors=100):
        from sklearn.neighbors import KNeighborsClassifier
        self.name = "KNN"
        self.tree = KNeighborsClassifier(n_neighbors=neighbors)

    def fit(self, X, y):
        self.tree.fit(X.T, y)

    def predict(self, X):
        y_hat = self.tree.predict(X.T)
        return y_hat
