import pickle
import numpy as np
from sklearn import tree

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.classifiers = list()
        self.a = list()
        pass

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self, X, y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        w = [1/len(X)]*len(X)
        for m in range(self.n_weakers_limit):
            print('start fitting the ' + str(m) + " classifier")
            self.classifiers.append(self.weak_classifier(max_depth=3))
            self.classifiers[m].fit(X, y, w)
            print('start predicting the ' + str(m) + " classifier")
            predicts = self.classifiers[m].predict(X)
            em = 0
            for i in range(len(predicts)):
                if predicts[i] != y[i][0]:
                    em += w[i]
            em = max(em, 1e-16)
            am = 0.5 * np.log((1-em) / em)
            self.a.append(am)
            zm = 0
            for i in range(len(X)):
                zm += w[i] * np.exp(-1 * am * (y[i][0]) * (predicts[i]))
            for i in range(len(X)):
                w[i] = (w[i]/zm)*np.exp(-1 * am * y[i][0] * predicts[i])

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        result = [[None] for i in range(len(X))]
        for index in range(len(X)):
            score = 0
            for m in range(self.n_weakers_limit):
                batch = [X[index]]
                _ = (self.classifiers[m].predict(batch))[0]
                score += self.a[m] * _
            result[index][0] = score
        return result
        pass

    def predict(self, X, threshold=0):
        '''Predict the catagories for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).

        '''
        scores = self.predict_scores(X)
        results = [[None] for i in range(len(scores))]
        for i in range(len(scores)):
            if scores[i][0] >= threshold:
                results[i][0] = 1
            else:
                results[i][0] = -1
        return results
        pass

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
